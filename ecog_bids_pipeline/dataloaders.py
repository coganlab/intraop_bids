from __future__ import annotations

from pathlib import Path
from tkinter import filedialog
import logging
import shutil
import subprocess
import numpy as np
from scipy.io import loadmat, wavfile
from scipy.signal import resample_poly, correlate, find_peaks
import noisereduce as nr
import matplotlib.pyplot as plt
import mne
import h5py
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from load_intan_rhd_format.load_intan_rhd_format import read_data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_TYPE = np.float64


class rhdLoader:

    def __init__(self,
                 subject: str,
                 rhd_dir: Optional[Union[str, Path]] = None,
                 out_dir: Optional[Union[str, Path]] = None,
                 fileIDs: Optional[Iterable[int]] = None,
                 array_type: Optional[str] = None) -> None:
        """Initialize an RHD data loader.

        Args:
            subject: Subject identifier string.
            rhd_dir: Path or string pointing to directory containing RHD files.
                If None, a file dialog will be opened to select a directory.
            out_dir: Output base directory where processed files will be
                written. If None, the default BIDS processing path is used.
            fileIDs: Optional iterable of integer indices specifying which
                RHD files to include (1-indexed by order). If None all files
                are used.
            array_type: Name of the electrode array channel map to use
                (e.g. '128-strip', '256-grid', '256-strip', or 'hybrid-strip').
                Default is None to prompt the user to select an array type.

        Returns:
            None
        """
        self.subject = subject
        if out_dir is None:
            out_dir = self._get_out_dir()
        self.out_dir = Path(out_dir)
        self._make_subj_dir(base_dir=self.out_dir)

        if rhd_dir is None:
            rhd_dir = filedialog.askdirectory(
                title=f'Select RHD data directory for subject {subject}')
        self.rhd_dir = Path(rhd_dir)
        self.fileIDs = fileIDs

        if array_type is None:
            arr_types = ['128-strip', '256-grid', '256-strip', 'hybrid-strip']
            types = '/'.join(arr_types)
            resp = input(f'Please specify array type {types}:')
            if resp in arr_types:
                array_type = resp
            else:
                raise ValueError('Invalid array type specified. This type may'
                                 'not be implemented yet.')
        self.channel_map = self._get_channel_map(array_type)

    def load_data(self) -> "rhdLoader":
        """Load RHD data for the initialized subject and save artifacts.

        This function finds the subject RHD files, reads them, computes bad
        channels based on impedance, saves the raw data as an HDF5 file, and
        writes additional miscellaneous arrays (trigger, mic, impedance) as
        NumPy files in the subject output directory.

        Returns:
            The loader instance (self) to allow method chaining.
        """
        fs_down = 2000  # target downsampled frequency in Hz
        logger.info(f'Loading RHD data for subject {self.subject} '
                    f'from directory: {self.rhd_dir}...')
        rhd_files = self._get_subj_files()
        print(f'Found files: {rhd_files}')
        all_data = self._build_full_rhd_data(rhd_files, fs_down)

        # calculate bad channels based on impedance
        bad_channels = self._get_high_impedance_channels(all_data['impedance'])
        all_data['bad_channels'] = bad_channels

        # save raw data as h5 file in output dir
        _ = self._save_raw_data(all_data['raw_data'],
                                fs_down,
                                all_data['bad_channels'],
                                self.channel_map,
                                f'sub-{self.subject}_raw.h5')

        # # see if we can free up memory here
        del all_data['raw_data']
        del all_data['bad_channels']

        # # save raw data to output dir
        # _ = self._save_mne_raw(raw, f'sub-{self.subject}_raw.edf')

        # save misc rhd data as numpy files in output dir
        _ = self._save_rhd_misc_data(all_data['trigger'],
                                     f'sub-{self.subject}_trigger.npy')
        _ = self._save_rhd_misc_data(all_data['mic'],
                                     f'sub-{self.subject}_mic.npy')
        _ = self._save_rhd_misc_data(all_data['impedance'],
                                     f'sub-{self.subject}_impedance.npy')
        self.fs = all_data['fs']

        logger.info('...data loading complete.')
        return self

    def make_cue_events(self) -> None:
        """Create cue event timings by cross-correlating stimulus audio.

        This method loads trial information, denoises the microphone trace,
        loads stimulus WAV files referenced by the trial information, detects
        trigger onsets, cross-correlates each stimulus with the mic trace to
        find precise stimulus onsets/offsets, and writes a
        `cue_events.txt` file in the subject output directory.

        Raises:
            FileNotFoundError: If the trialInfo.mat file is not found in the
                RHD data directory.
        """
        xcorr_start_buffer = 0.5  # seconds
        xcorr_end_buffer = 0.7  # seconds

        trialInfo_path = self.rhd_dir / 'trialInfo.mat'
        if not trialInfo_path.exists():
            logger.error(f'trialInfo.mat file not found in {self.rhd_dir}')
            raise FileNotFoundError('trialInfo.mat file not found in '
                                    f'{self.rhd_dir}')
        trial_info = loadmat(trialInfo_path)['trialInfo'].squeeze()

        mic = np.load(self.out_dir / f'sub-{self.subject}' / ('sub-' +
                      self.subject + '_mic.npy'))

        fs = self._get_fs()
        mic_denoised = nr.reduce_noise(y=mic, sr=fs,
                                       stationary=False, prop_decrease=0.9)

        stim_files = self._get_stim_dir().rglob('**/*.wav')

        # === Relevant stimuli from trial info ===
        rel_stims = [t["sound"][0, 0][0] for t in trial_info]
        rel_stims_set = set(rel_stims)
        # Filter only relevant stimuli
        rel_stims_info = [(f.parent, f.name) for f in stim_files if f.name in
                          rel_stims_set]

        # Remove duplicates by filename
        seen = set()
        rel_stims_info_unique = []
        for folder, name in rel_stims_info:
            if name not in seen:
                rel_stims_info_unique.append((folder, name))
                seen.add(name)

        # === Load all stimuli ===
        stims = {}
        for stim_folder, stim_name in rel_stims_info_unique:
            stim_path = stim_folder / stim_name
            fs_stim, stim = wavfile.read(stim_path)
            stim = stim[:, 0] if stim.ndim > 1 else stim
            # Resample to fs of main audio
            stim_resamp = resample_poly(stim, fs, fs_stim)
            stim_name_no_ext = Path(stim_name).stem
            stims[stim_name_no_ext] = stim_resamp

        trigger = np.load(self.out_dir / f'sub-{self.subject}' / ('sub-' +
                          self.subject + '_trigger.npy'))
        trig_times = self.detect_trigger_onsets(trigger, fs)

        # === Process trials ===
        with open(self.out_dir / f'sub-{self.subject}' / 'cue_events.txt',
                  'w') as fid_cue:
            for iTrial in range(len(trial_info)):
                curr_stim = rel_stims[iTrial]
                curr_stim_name = Path(curr_stim).stem
                curr_stim_aud = stims[curr_stim_name]

                on = trig_times[iTrial]

                # Define cross-correlation search window
                xcorr_start = int(max((on - xcorr_start_buffer) * fs, 0))
                xcorr_end = int(min((on + xcorr_end_buffer) * fs,
                                    len(mic_denoised)))
                allblocks_win = mic_denoised[xcorr_start:xcorr_end]

                # Cross-correlation
                xcorr_val = correlate(allblocks_win, curr_stim_aud,
                                      mode="full")
                lags = np.arange(-len(curr_stim_aud) + 1, len(allblocks_win))
                lag_idx = np.argmax(np.abs(xcorr_val))
                l_offset = lags[lag_idx]
                stim_onset = (xcorr_start + l_offset) / fs

                # Duration and offset
                dur = len(curr_stim_aud) / fs
                stim_offset = stim_onset + dur

                fid_cue.write(f"{stim_onset:.6f}\t{stim_offset:.6f}\t"
                              f"{iTrial+1}_{curr_stim_name}\n")
        logger.info('Successfully created cue_events.txt')

    @staticmethod
    def detect_trigger_onsets(trigger: np.ndarray,
                              fs: float,
                              min_dist_s: float = 3.0,
                              threshold_factor: float = 5.0,
                              task_start_buffer_s: float = 3.0,
                              peak_rejection_factor: float = 7.0,
                              interactive: bool = True) -> np.ndarray:
        """
        Automatically detect trial trigger onsets from a photodiode trace.

        The function thresholds the trigger trace to find a wide "task
        start" pulse and then detects individual trial peaks after that
        time. A figure is plotted and the user may optionally override
        detection parameters interactively.

        Args:
            trigger: 1D numpy array containing the photodiode/trigger trace
                voltage samples.
            fs: Sampling rate of the trigger trace in Hz.
            min_dist_s: Minimum allowed distance between detected trials in
                seconds.
            threshold_factor: Multiplier of the post-task standard deviation to
                set the trial detection threshold.
            task_start_buffer_s: Seconds to offset the detected task start
                peak to ensure the search window excludes the wide pulse.
            peak_rejection_factor: Multiplier of peak-height SD used to reject
                amplitude outliers.
            interactive: If True, show plots and permit manual overrides via
                the console.

        Returns:
            A 1D numpy array of detected trial onset times in seconds.
        """

        trig = np.asarray(trigger, dtype=float)
        t_axis = np.arange(len(trig)) / fs

        # --- Step 1: Rough normalization ---
        baseline = np.percentile(trig, 10)
        trig_z = trig - baseline
        std_est = np.std(trig_z)
        task_start_thresh = baseline + threshold_factor * std_est

        trig = np.maximum(trig, baseline)

        # --- Step 2: Find the wide pulse for task start ---
        peaks, props = find_peaks(trig, height=task_start_thresh,
                                  width=int(0.5*fs))
        print(f"Detected {len(peaks)} potential task start pulses.")
        if len(peaks) == 0:
            print("No task start pulse detected automatically.")
            task_start_idx = 0
        else:
            # use last peak in case task had to be restarted
            task_start_idx = peaks[-1] + int(task_start_buffer_s * fs)

        task_start_time = task_start_idx / fs

        # --- Step 3: Define search window for trials ---
        trig_post = trig[task_start_idx:]
        baseline_post = np.percentile(trig_post, 10)
        std_post = np.std(trig_post - baseline_post)
        threshold = baseline_post + threshold_factor * std_post

        min_samples = int(min_dist_s * fs)
        peaks_trials, props_trials = find_peaks(trig_post, height=threshold,
                                                distance=min_samples)
        trial_onsets = (peaks_trials + task_start_idx) / fs

        # --- Step 4: Look for outliers in amplitudes of trigger peaks ---
        if len(peaks_trials) > 0:
            peak_heights = props_trials['peak_heights']
            mean_height = np.mean(peak_heights)
            std_height = np.std(peak_heights)
            valid_idxs = np.where(np.abs(peak_heights - mean_height) <=
                                  peak_rejection_factor * std_height)[0]
            trial_onsets = trial_onsets[valid_idxs]

        print(f"Detected {len(trial_onsets)} "
              f"trials after t={task_start_time:.2f}s.")

        # --- Step 5: Plot results ---
        plt.figure(figsize=(12, 4))
        plt.plot(t_axis, trig, label="Trigger")
        plt.axhline(threshold, color='r', linestyle='--',
                    label=f"Trial threshold = {threshold:.3f}")
        plt.axvline(task_start_time, color='g', linestyle='--',
                    label="Task start")
        plt.plot(trial_onsets, trig[(trial_onsets * fs).astype(int)], 'rx',
                 label="Detected trials")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (V)")
        plt.title("Automatic Trigger Detection")
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show(block=False)

        # --- Step 6: Optional manual override ---
        if interactive:
            resp = ''
            while resp not in ['yes', 'y', 'quit', 'q']:
                resp = input("Are these detections correct? "
                             "(yes-y/no-n/quit-q): ").strip().lower()
                if resp in ['n', 'no']:
                    start_time = float(input("Enter start time (s): "))
                    end_time = float(input("Enter end time (s): "))
                    threshold = float(input("Enter threshold (V): "))
                    min_dist_s = float(input("Enter min distance (s): "))

                    # Re-run detection with user inputs
                    start_idx = int(start_time * fs)
                    end_idx = int(end_time * fs)
                    trig_crop = trig[start_idx:end_idx]
                    peaks_manual, _ = find_peaks(trig_crop, height=threshold,
                                                 distance=int(min_dist_s*fs))
                    trial_onsets = (peaks_manual + start_idx) / fs

                    print(f"✅ {len(trial_onsets)} trials detected manually.")

                    plt.figure(figsize=(12, 4))
                    plt.plot(t_axis, trig, label="Trigger")
                    plt.plot(trial_onsets, trig[(trial_onsets *
                                                 fs).astype(int)], 'rx')
                    plt.axvline(start_time, color='g', linestyle='--')
                    plt.axvline(end_time, color='g', linestyle='--')
                    plt.axhline(threshold, color='r', linestyle='--')
                    plt.xlabel("Time (s)")
                    plt.ylabel("Amplitude (V)")
                    plt.title("Manual Trigger Detection")
                    plt.show(block=False)
                elif resp not in ['yes', 'y', 'quit', 'q']:
                    print("Invalid response. Please enter yes, no, or quit.")

        return trial_onsets

    def run_mfa(self, task_name: str = 'phoneme_sequencing') -> bool:
        """Run the Montreal Forced Aligner (MFA) pipeline for this subject.

        This method copies trialInfo.mat to the subject output directory,
        writes the microphone trace to a WAV file, then invokes the external
        MFA pipeline script. If MFA completes successfully, MFA-produced
        token files are moved into the subject directory.

        Args:
            task_name: Name of the MFA task configuration to run (keys in
                the `MFA_pipeline/conf/task` folder).

        Returns:
            True if MFA ran and processed successfully, False otherwise.
        """
        # copy trialInfo to subject directory
        trialInfo_src = self.rhd_dir / 'trialInfo.mat'
        trialInfo_dst = self.out_dir / f'sub-{self.subject}' / 'trialInfo.mat'
        shutil.copyfile(trialInfo_src, trialInfo_dst)

        # save microphone data as wav file in subject directory
        mic = np.load(self.out_dir / f'sub-{self.subject}' / ('sub-' +
                      self.subject + '_mic.npy'))
        mic_wav_path = self.out_dir / f'sub-{self.subject}' / 'allblocks.wav'
        fs = self._get_fs()
        wavfile.write(mic_wav_path, int(fs), mic.astype(DATA_TYPE))

        # run MFA (assumes MFA is cloned to
        # Box/CoganLab/Data/Micro/BIDS_processing/MFA_pipeline)
        mfa_cmd = ['python',
                   str(Path(self._get_out_dir() / 'MFA_pipeline' /
                       'mfa_pipeline.py')),
                   'task=' + task_name,
                   'patient_dir=' + str(self.out_dir),
                   'patients=sub-' + self.subject,
                   ]
        try:
            logger.info(f'Running MFA for subject {self.subject}...')
            subprocess.run(mfa_cmd, check=True)
            logger.info('...MFA processing complete.')
            self._process_mfa_output()
        except subprocess.CalledProcessError as e:
            logger.error(f'Error running MFA: {e}')
            return False
        return True

    def create_trials_dict(self) -> Dict[str, List[Any]]:
        """Create a trials dictionary from MFA token-level annotation files.

        Reads the token-level word timing files produced by MFA for stimulus
        and response channels and assembles a dictionary with lists for
        stimulus words and onset/offset times.

        Returns:
            A dictionary with keys: 'stimulus', 'auditory_onset',
            'auditory_offset', 'response_onset', 'response_offset'. Each
            value is a list corresponding to trials in order.
        """
        trials_dict = {
            'stimulus': [],
            'auditory_onset': [],
            'auditory_offset': [],
            'response_onset': [],
            'response_offset': [],
        }
        # open MFA token-level annotations to read in parallel
        fname_stim = (
            self.out_dir / f'sub-{self.subject}' / 'mfa_stim_words.txt'
        )
        f_stim = open(fname_stim, 'r')
        fname_resp = (
            self.out_dir / f'sub-{self.subject}' / 'mfa_resp_words.txt'
        )
        f_resp = open(fname_resp, 'r')

        # save timings for each trial to trials_dict
        for line_stim, line_resp in zip(f_stim, f_resp):
            stim_onset, stim_offset, stim_word = line_stim.strip().split('\t')
            resp_onset, resp_offset, resp_word = line_resp.strip().split('\t')

            trials_dict['stimulus'].append(stim_word)
            trials_dict['auditory_onset'].append(float(stim_onset))
            trials_dict['auditory_offset'].append(float(stim_offset))
            trials_dict['response_onset'].append(float(resp_onset))
            trials_dict['response_offset'].append(float(resp_offset))
        f_stim.close()
        f_resp.close()
        return trials_dict

    def _get_subj_files(self) -> List[Path]:
        """Return a sorted list of RHD files for the subject.

        Returns:
            A list of Path objects matching the subject pattern in the RHD
            directory. If `self.fileIDs` is provided, the returned list will
            be filtered to those indices (1-indexed by order).
        """
        pattern = f'{self.subject}*.rhd'
        rhd_files = sorted(self.rhd_dir.glob(pattern))
        # only keep files selected by numerical IDs corresponding to order
        if self.fileIDs is not None:
            logger.info(f'Using RHD files: {self.fileIDs}')
            rhd_files = [f for i, f in enumerate(rhd_files) if i+1 in
                         self.fileIDs]
        logger.info(f'Found files: {[f.name for f in rhd_files]}')
        return rhd_files

    def _get_channel_map(self, array_type: str) -> np.ndarray:
        """Load a channel map .mat file for the specified array type.

        Args:
            array_type: Name of the array channel map file (without .mat).

        Returns:
            A numpy array containing the channel map loaded from the .mat
            file.
        """
        # load channel map based on array type
        chan_map_dir = Path(__file__).parent / 'channel_maps'
        chan_map_fname = chan_map_dir / f'{array_type}.mat'
        chan_map = loadmat(chan_map_fname)['chanMap'].squeeze()
        return chan_map

    def _build_full_rhd_data(
        self, rhd_files: Sequence[Path], fs_down: float
    ) -> Dict[str, Any]:
        """Read and concatenate data from multiple RHD files.

        Args:
            rhd_files: Sequence of Path objects pointing to .rhd files for the
                subject (typically each file is ~1 minute of data).

        Returns:
            A dictionary with keys: 'raw_data' (2D numpy array channels x
            samples), 'trigger', 'mic', 'impedance' (1D array), and 'fs'.
        """
        n_chans = np.nanmax(self.channel_map).astype(int)  # 1-indexed chans
        amplifier_data_all = np.empty((n_chans, 0), dtype=DATA_TYPE)
        trigger_all = np.empty((0,), dtype=DATA_TYPE)
        mic_all = np.empty((0,), dtype=DATA_TYPE)
        impedance_all = np.empty((0, n_chans), dtype=DATA_TYPE)
        # store data from each RHD file (corresponds to 1 min of data)
        log_downsample = False
        for rhd_file in rhd_files:
            logger.info(f'Loading RHD file: {rhd_file.name}')
            data = read_data(rhd_file)  # resample raw data to 2 kHz

            fs_data = data['frequency_parameters']['amplifier_sample_rate']
            if fs_data != fs_down:
                if not log_downsample:
                    logger.info(f'Resampling raw data from {fs_data} Hz to '
                                f'{fs_down} Hz...')
                    log_downsample = True
                amp_data = resample_poly(data['amplifier_data'],
                                         fs_down, fs_data,
                                         axis=1)
            else:
                amp_data = data['amplifier_data']
            amplifier_data_all = np.append(
                amplifier_data_all,
                amp_data.astype(DATA_TYPE),
                axis=1)
            trigger_all = np.append(
                trigger_all,
                data['board_adc_data'][0, :].astype(DATA_TYPE))
            mic_all = np.append(
                mic_all,
                data['board_adc_data'][1, :].astype(DATA_TYPE))
            impedance_all = np.append(
                impedance_all,
                self._get_impedance_magnitudes(data)[
                    np.newaxis, :].astype(DATA_TYPE),
                axis=0,
            )

            logger.info(f'      Data shape: {data["amplifier_data"].shape}')
            logger.info(
                f'      Trigger shape: {data["board_adc_data"][0, :].shape}'
            )
            logger.info(
                f'      Mic shape: {data["board_adc_data"][1, :].shape}'
            )
            logger.info(f'      Impedance shape: '
                        f'{impedance_all[-1].shape}')

        # save impedance as mean across files (don't know if this changes per
        # file or not but if it doesn't mean is still ok)
        impedance = np.array(impedance_all).mean(axis=0)
        full_data = {
            'raw_data': amplifier_data_all,
            'trigger': trigger_all,
            'mic': mic_all,
            'impedance': impedance,
            'fs': data['frequency_parameters']['amplifier_sample_rate'],
        }
        return full_data

    def _convert_to_mne_raw(self,
                            amplifier_data: np.ndarray,
                            fs: float,
                            bad_channels: Sequence[int],
                            units: str = 'uV') -> mne.io.BaseRaw:
        """Convert raw amplifier data to an MNE Raw object.

        Args:
            amplifier_data: 2D numpy array of shape (n_channels, n_samples)
                containing amplifier voltage traces.
            fs: Sampling frequency in Hz.
            bad_channels: Sequence of 1-indexed channel numbers to mark as
                bad in the returned Raw object.
            units: Unit string for input data. Supported units: 'V', 'mV',
                'uV', 'nV'. The data will be scaled accordingly.

        Returns:
            An MNE Raw object containing the converted data and channel info.
        """
        match units:
            case "V":
                factor = 1
            case "mV":
                factor = 1e-3
            case "uV":
                factor = 1e-6
            case "nV":
                factor = 1e-9
            case _:
                raise NotImplementedError("Unit " + units +
                                          " not implemented yet")
        channels = self.channel_map.flatten()  # get 1D list of channels
        # remove NaN channels (no channels here, filler for rectangular map)
        channels = channels[~np.isnan(channels)].astype(int)
        # remove duplicates, possible for hybrid arrays where macro channels
        # take the space of multiple micro channels
        channels = np.unique(channels).astype(str).tolist()

        info = mne.create_info(channels, sfreq=fs, ch_types='ecog')
        info['bads'] = [str(c) for c in bad_channels]
        raw = mne.io.RawArray(amplifier_data * factor, info)
        return raw

    def _save_mne_raw(
        self, raw: mne.io.BaseRaw, fname: Union[str, Path]
    ) -> bool:
        """Export an MNE Raw object to a file.

        Args:
            raw: MNE Raw object to export.
            fname: Destination filename (string or Path).

        Returns:
            True if saving succeeded, False if an exception occurred.
        """
        save_path = self.out_dir / f'sub-{self.subject}' / fname
        logger.info(f'Saving MNE Raw data to {save_path}...')
        try:
            raw.export(save_path)
            logger.info('...MNE Raw data saved successfully.')
            return True
        except Exception as e:
            logger.error(f'Error saving MNE Raw data: {e}')
            print('Failed to save MNE Raw data. Error written to log.')
            return False

    def _save_raw_data(self,
                       data: np.ndarray,
                       fs: float,
                       bad_channels: Sequence[int],
                       channel_map: np.ndarray,
                       fname: Union[str, Path]) -> bool:
        """Save raw data and metadata to an HDF5 file.

        Args:
            data: 2D numpy array of raw amplifier data (channels x samples).
            fs: Sampling frequency.
            bad_channels: Sequence of channel indices marked as bad.
            channel_map: Channel map array saved alongside the data.
            fname: Filename for the HDF5 container to create in the subject
                directory.

        Returns:
            True if the HDF5 file was written successfully, False otherwise.
        """
        save_path = self.out_dir / f'sub-{self.subject}' / fname
        logger.info(f'Saving raw data to {save_path}...')
        try:
            with h5py.File(save_path, 'w') as hf:
                hf.create_dataset('data', data=data)
                hf.create_dataset('fs', data=fs)
                hf.create_dataset('bad_channels', data=bad_channels)
                hf.create_dataset('channel_map', data=channel_map)
            logger.info('...raw data saved successfully.')
            return True
        except Exception as e:
            logger.error(f'Error saving raw data: {e}')
            print('Failed to save raw data. Error written to log.')
            return False

    def _save_rhd_misc_data(
        self, data: np.ndarray, fname: Union[str, Path]
    ) -> bool:
        """Save miscellaneous RHD arrays as numpy .npy files.

        Args:
            data: 1D or 2D numpy array to save.
            fname: Filename under the subject output directory.

        Returns:
            True on success, False on failure.
        """
        save_path = self.out_dir / f'sub-{self.subject}' / fname
        logger.info(f'Saving RHD misc data to {save_path}...')
        try:
            np.save(save_path, data)
            logger.info('...RHD misc data saved successfully.')
            return True
        except Exception as e:
            logger.error(f'Error saving RHD misc data: {e}')
            print('Failed to save RHD misc data. Error written to log.')
            return False

    def _get_impedance_magnitudes(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract impedance magnitude values from raw RHD metadata.

        Args:
            data: Dictionary-like structure returned by the RHD reader which
                contains an 'amplifier_channels' entry.

        Returns:
            1D numpy array of impedance magnitudes (one per channel).
        """
        impedance_mag = np.array([
            data['amplifier_channels'][iChan]['electrode_impedance_magnitude']
            for iChan in range(len(data['amplifier_channels']))
        ])
        return impedance_mag

    def _get_high_impedance_channels(
        self, impedance: np.ndarray, threshold: float = 1e6
    ) -> List[int]:
        """Return list of 1-indexed channels with impedance above threshold.

        Args:
            impedance: 1D array of impedance magnitudes.
            threshold: Threshold above which a channel is considered bad.

        Returns:
            List of 1-indexed channel numbers whose impedance exceeds the
            threshold.
        """
        bad_channels = np.where(impedance > threshold)[0] + 1  # 1-indexed
        return bad_channels.tolist()

    def _get_fs(self) -> float:
        """Return the sampling frequency for the stored or saved data.

        If `self.fs` is already set it is returned. Otherwise the method
        attempts to read a saved EDF file in the subject directory to infer
        the sampling rate.

        Returns:
            Sampling frequency in Hz.

        Raises:
            ValueError: If sampling frequency cannot be determined.
        """
        if hasattr(self, 'fs'):
            return self.fs
        elif (self.out_dir / f'sub-{self.subject}' /
              f'sub-{self.subject}_raw.h5').exists():
            with h5py.File(self.out_dir / f'sub-{self.subject}' /
                           f'sub-{self.subject}_raw.h5', 'r') as hf:
                self.fs = hf['fs'][()]
            return self.fs
        elif (self.out_dir / f'sub-{self.subject}' /
              f'sub-{self.subject}_raw.edf').exists():
            raw = mne.io.read_raw_edf(
                self.out_dir / f'sub-{self.subject}' /
                f'sub-{self.subject}_raw.edf', preload=False)
            self.fs = raw.info['sfreq']
            return raw.info['sfreq']
        else:
            raise ValueError('Sampling frequency not found. Please load data'
                             ' first to determine fs.')

    def _process_mfa_output(self) -> None:
        """Move MFA output token files from MFA subdirectory into subject dir.

        The function attempts to move a small set of known MFA output files
        (stim/resp words and phones). Missing files are logged as warnings.
        """
        subj_dir = self.out_dir / f'sub-{self.subject}'
        mfa_dir = subj_dir / 'mfa'
        files_to_move = ['mfa_stim_words.txt', 'mfa_stim_phones.txt',
                         'mfa_resp_words.txt', 'mfa_resp_phones.txt']
        for file_name in files_to_move:
            src_path = mfa_dir / file_name
            dst_path = subj_dir / file_name
            if src_path.exists():
                shutil.move(src_path, dst_path)
                logger.info(f'Moved {file_name} to subject directory.')
            else:
                logger.warning(f'{file_name} not found in MFA directory.')

    def _get_out_dir(self) -> Path:
        """Return the default BIDS processing output directory under Box.

        The location is determined relative to the current user's home
        directory. This method does not validate that the returned path
        actually exists beyond being a Path object.
        """
        # get path to BIDS processing folder on Box
        user_home = Path.home()
        box_dir = user_home / r'Box\CoganLab\Data\Micro\BIDS_processing'
        return box_dir

    def _get_stim_dir(self) -> Path:
        """Return the default stimulus directory under the user's Box folder.

        Returns:
            A Path pointing to the stimulus folder used to find WAV files
            referenced by trial information.
        """
        user_home = Path.home()
        stim_dir = user_home / r'Box\CoganLab\ECoG_Task_Data\Stim'
        return stim_dir

    def _make_subj_dir(self, base_dir: Union[str, Path]) -> None:
        """Create the subject directory and attach a file logging handler.

        Args:
            base_dir: Base directory under which to create the subject folder.
        """
        # create directiry for subject if it doesn't exist
        subj_dir = Path(base_dir) / f'sub-{self.subject}'
        subj_dir.mkdir(parents=True, exist_ok=True)

        # create new logging file in subject directory or append to existing
        log_file = subj_dir / 'BIDS_processing.log'
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - '
                                      '%(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info('-' * 100)
        logger.info(f'Subject directory set to: {subj_dir}')


def main(data_dir, subject, fileIDs, array_type='None',
         task='lexical_repeat_intraop'):
    loader = rhdLoader(subject, data_dir, fileIDs=fileIDs,
                       array_type=array_type)
    loader.load_data()
    loader.make_cue_events()
    loader.run_mfa(task_name=task)
    print(loader.create_trials_dict())
    print('RHD data loaded successfully.')


if __name__ == "__main__":
    user_path = Path.home()
    # data_dir = None

    # data_dir = (user_path /
    #             r'Box\CoganLab\uECoG_Upload\S26_04_20_2021_Human_Intraop')
    # subject = 'S26'
    # fileIDs = range(31, 43)
    # main(data_dir, subject, fileIDs, array_type='128-strip',
    #      task='phoneme_sequencing')

    # data_dir = (user_path /
    #             r'Box\CoganLab\uECoG_Upload\S78_08_20_2025'
    #             r'\S78_IntraOp_250820_111305')
    # subject = 'S78'
    # fileIDs = None
    # main(data_dir, subject, fileIDs, array_type='256-grid',
    #      task='lexical_repeat_intraop')

    # data_dir = (user_path /
    #             r'Box\CoganLab\uECoG_Upload\S73_03_18_2025'
    #             r'\S73_V1_250318_120342')
    # subject = 'S73'
    # fileIDs = None
    # main(data_dir, subject, fileIDs, array_type='256-strip',
    #      task='lexical_repeat_intraop')

    data_dir = (user_path /
                r'C:\Users\zms14\Box\CoganLab\uECoG_Upload\S57_12_19_2023'
                r'\S57_Recording_231219_113044')
    subject = 'S57'
    fileIDs = None
    main(data_dir, subject, fileIDs, array_type='hybrid-strip',
         task='phoneme_sequencing')
