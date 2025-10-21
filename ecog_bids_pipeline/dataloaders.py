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

from load_intan_rhd_format.load_intan_rhd_format import read_data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class rhdLoader:

    def __init__(self, subject, rhd_dir=None, out_dir=None, fileIDs=None,
                 array_type='None'):
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
            resp = input(f'Please specify array type {'/'.join(arr_types)}:')
            if resp in arr_types:
                array_type = resp
            else:
                raise ValueError('Invalid array type specified. This type may'
                                 'not be implemented yet.')
        self.channel_map = self._get_channel_map(array_type)

    def load_data(self):
        logger.info(f'Loading RHD data for subject {self.subject} '
                    f'from directory: {self.rhd_dir}...')
        rhd_files = self._get_subj_files()
        print(f'Found files: {rhd_files}')
        all_data = self._build_full_rhd_data(rhd_files)

        # convert amplifier data to MNE Raw object
        raw = self._convert_to_mne_raw(all_data['raw_data'],
                                       all_data['fs'])
        all_data['raw_data'] = raw

        # calculate bad channels based on impedance
        bad_channels = self._get_high_impedance_channels(all_data['impedance'])
        all_data['bad_channels'] = bad_channels

        # save data as object attributes
        for k, v in all_data.items():
            self.__setattr__(k, v)
        logger.info('...data loading complete.')
        print('Successfully loaded RHD data to rhdLoader object attributes.')

        return self

    def make_cue_events(self):
        xcorr_start_buffer = 0.5  # seconds
        xcorr_end_buffer = 0.7  # seconds

        trialInfo_path = self.rhd_dir / 'trialInfo.mat'
        if not trialInfo_path.exists():
            logger.error(f'trialInfo.mat file not found in {self.rhd_dir}')
            raise FileNotFoundError('trialInfo.mat file not found in '
                                    f'{self.rhd_dir}')
        trial_info = loadmat(trialInfo_path)['trialInfo'].squeeze()

        mic_denoised = nr.reduce_noise(y=self.mic, sr=self.fs,
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
            stim_resamp = resample_poly(stim, self.fs, fs_stim)
            stim_name_no_ext = Path(stim_name).stem
            stims[stim_name_no_ext] = stim_resamp

        trig_times = self.detect_trigger_onsets(self.trigger, self.fs)

        # === Process trials ===
        with open(self.out_dir / f'sub-{self.subject}' / 'cue_events.txt',
                  'w') as fid_cue:
            for iTrial in range(len(trial_info)):
                curr_stim = rel_stims[iTrial]
                curr_stim_name = Path(curr_stim).stem
                curr_stim_aud = stims[curr_stim_name]

                on = trig_times[iTrial]

                # Define cross-correlation search window
                xcorr_start = int(max((on - xcorr_start_buffer) * self.fs, 0))
                xcorr_end = int(min((on + xcorr_end_buffer) * self.fs,
                                    len(mic_denoised)))
                allblocks_win = mic_denoised[xcorr_start:xcorr_end]

                # Cross-correlation
                xcorr_val = correlate(allblocks_win, curr_stim_aud,
                                      mode="full")
                lags = np.arange(-len(curr_stim_aud) + 1, len(allblocks_win))
                lag_idx = np.argmax(np.abs(xcorr_val))
                l_offset = lags[lag_idx]
                stim_onset = (xcorr_start + l_offset) / self.fs

                # Duration and offset
                dur = len(curr_stim_aud) / self.fs
                stim_offset = stim_onset + dur

                fid_cue.write(f"{stim_onset:.6f}\t{stim_offset:.6f}\t"
                              f"{iTrial+1}_{curr_stim_name}\n")
        logger.info('Successfully created cue events.txt')

    @staticmethod
    def detect_trigger_onsets(trigger, fs,
                              min_dist_s=3,
                              threshold_factor=5.0,
                              task_start_buffer_s=3.0,
                              peak_rejection_factor=7.0,
                              interactive=True):
        """
        Automatically detect trial triggers from a photodiode trace.

        Parameters
        ----------
        trigger : np.ndarray
            Trigger voltage trace.
        fs : float
            Sampling rate (Hz).
        min_dist_s : float
            Minimum distance between detected trials (s).
        start_threshold_factor : float
            Std multiplier to find the large 'start of task' pulse.
        trial_threshold_factor : float
            Std multiplier for normal trial detection threshold.
        post_buffer_s : float
            How long after last detected trial to ignore (to skip teardown).
        interactive : bool
            If True, shows results and allows user override.

        Returns
        -------
        trial_onsets : np.ndarray
            Times (s) of detected trial triggers.
        task_start_time : float
            Time (s) of detected task start.
        threshold : float
            Threshold used for trial detection.
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

    def run_mfa(self, task_name='phoneme_sequencing'):
        # copy trialInfo to subject directory
        trialInfo_src = self.rhd_dir / 'trialInfo.mat'
        trialInfo_dst = self.out_dir / f'sub-{self.subject}' / 'trialInfo.mat'
        shutil.copyfile(trialInfo_src, trialInfo_dst)

        # save microphone data as wav file in subject directory
        mic_wav_path = self.out_dir / f'sub-{self.subject}' / 'allblocks.wav'
        wavfile.write(mic_wav_path, int(self.fs), self.mic.astype(np.float32))

        # run MFA (assumes MFA is cloned to
        # Box/CoganLab/Data/Micro/BIDS_processing/MFA_pipeline)
        mfa_cmd = ['python',
                   Path(self._get_out_dir() / 'MFA_pipeline' /
                        'mfa_pipeline.py'),
                   'task=' + task_name,
                   'patient_dir=' + str(self.out_dir),
                   'patients=sub-' + self.subject,
                   'debug_mode=True',
                   ]
        try:
            logger.info(f'Running MFA for subject {self.subject}...')
            subprocess.run(mfa_cmd, check=True, shell=True)
            logger.info('MFA processing complete.')
            self._process_mfa_output()
        except subprocess.CalledProcessError as e:
            logger.error(f'Error running MFA: {e}')
            return False
        return True

    def create_trials_dict(self):
        trials_dict = {
            'stimulus': [],
            'auditory_onset': [],
            'auditory_offset': [],
            'response_onset': [],
            'response_offset': [],
        }
        # open MFA token-level annotations to read in parallel
        fname_stim = self.out_dir / f'sub-{self.subject}' / 'mfa_stim_words.txt'
        f_stim = open(fname_stim, 'r')
        fname_resp = self.out_dir / f'sub-{self.subject}' / 'mfa_resp_words.txt'
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

    def _get_subj_files(self):
        pattern = f'{self.subject}*.rhd'
        rhd_files = sorted(self.rhd_dir.glob(pattern))
        # only keep files selected by numerical IDs corresponding to order
        if self.fileIDs is not None:
            logger.info(f'Using RHD files: {self.fileIDs}')
            rhd_files = [f for i, f in enumerate(rhd_files) if i+1 in
                         self.fileIDs]
        logger.info(f'Found files: {[f.name for f in rhd_files]}')
        return rhd_files

    def _get_channel_map(self, array_type):
        # load channel map based on array type
        chan_map_dir = Path(__file__).parent / 'channel_maps'
        chan_map_fname = chan_map_dir / f'{array_type}.mat'
        chan_map = loadmat(chan_map_fname)['chanMap'].squeeze()
        return chan_map

    def _build_full_rhd_data(self, rhd_files):
        amplifier_data_all = []
        trigger_all = []
        mic_all = []
        impedance_all = []
        # store data from each RHD file (corresponds to 1 min of data)
        for rhd_file in rhd_files:
            logger.info(f'Loading RHD file: {rhd_file.name}')
            data = read_data(rhd_file)
            amplifier_data_all.append(data['amplifier_data'])
            logger.info(f'      Data shape: {data["amplifier_data"].shape}')
            trigger_all.append(data['board_adc_data'][0, :])
            logger.info(f'      Trigger shape: {data["board_adc_data"][0, :].shape}')
            mic_all.append(data['board_adc_data'][1, :])
            logger.info(f'      Mic shape: {data["board_adc_data"][1, :].shape}')
            impedance_all.append(self._get_impedance_magnitudes(data))
            logger.info(f'      Impedance shape: '
                        f'{impedance_all[-1].shape}')
        # combine to single array per data type
        amplifier_data_all = np.concatenate(amplifier_data_all, axis=1)
        trigger_all = np.concatenate(trigger_all)
        mic_all = np.concatenate(mic_all)

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

    def _convert_to_mne_raw(self, amplifier_data, fs, units='uV'):
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
        channels = channels[~np.isnan(channels)] 
        # remove duplicates, possible for hybrid arrays where macro channels
        # take the space of multiple micro channels
        channels = np.unique(channels)
        channels = np.sort(channels).astype(str).tolist()

        info = mne.create_info(channels, sfreq=fs, ch_types='ecog')
        raw = mne.io.RawArray(amplifier_data * factor, info)
        return raw

    def _get_impedance_magnitudes(self, data):
        impedance_mag = np.array([
            data['amplifier_channels'][iChan]['electrode_impedance_magnitude']
            for iChan in range(len(data['amplifier_channels']))
        ])
        return impedance_mag

    def _get_high_impedance_channels(self, impedance, threshold=1e6):
        return np.where(impedance > threshold)[0].tolist()

    def _process_mfa_output(self):
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

    def _get_out_dir(self):
        # get path to BIDS processing folder on Box
        user_home = Path.home()
        box_dir = user_home / r'Box\CoganLab\Data\Micro\BIDS_processing'
        return box_dir

    def _get_stim_dir(self):
        user_home = Path.home()
        stim_dir = user_home / r'Box\CoganLab\ECoG_Task_Data\Stim'
        return stim_dir

    def _make_subj_dir(self, base_dir):
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


def main(data_dir, subject, fileIDs, array_type='None', task='lexical_repeat_intraop'):
    loader = rhdLoader(subject, data_dir, fileIDs=fileIDs,
                       array_type=array_type)
    loader.load_data()
    loader.make_cue_events()
    loader.run_mfa(task_name=task)
    loader.create_trials_dict()
    print('RHD data loaded successfully.')


if __name__ == "__main__":
    data_dir = r'C:\Users\zms14\Box\CoganLab\uECoG_Upload\S26_04_20_2021_Human_Intraop'
    # data_dir = None
    subject = 'S26'
    fileIDs = range(31, 43)
    # fileIDs = None
    main(data_dir, subject, fileIDs, array_type='128-strip', task='phoneme_sequencing')
