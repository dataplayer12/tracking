import numpy as np
import os
import sys
import cv2
from cython_modules import lfit_cython
import csv
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot
from bokeh.io import export_png
from scipy.io import wavfile
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
from collections import Counter
from config import * #FPS, F_0, AUDIO_RATE, FOURCC, FIND_OSCILLATING, NUM_FRAMES_IN_HISTORY, MAX_KALMAN_LEARNING_TIME

class MovingObj:

    def __init__(self, center):
        self.previous_centers = [center]
        self.kalman = self.prepareKF()
        self.updateKF()
        self.num_frames_detected = 1
        self.num_not_found = 0
        self.is_being_tracked = False
        self.tracked_frame_indices = []
        self.is_oscillating = False

    def prepareKF(self):
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = 0.3 * np.eye(4).astype(np.float32)
        kalman.measurementNoiseCov = 0.3 * np.eye(2).astype(np.float32)
        return kalman

    def updateKF(self):
        self.kalman.correct(
            np.array(self.previous_centers[-1], dtype=np.float32))

    def firstcenter(self):
        return self.previous_centers[0]

    def lastcenter(self):
        return self.previous_centers[-1]

    def predictnow(self):
        if self.num_frames_detected < MAX_KALMAN_LEARNING_TIME or not self.is_being_tracked:
            if self.num_frames_detected > NUM_FRAMES_IN_HISTORY:
                #linear extrapolation
                pos = 2 * \
                    np.array(self.previous_centers[-1]) - \
                    np.array(self.previous_centers[-2])
                return list(pos) 

            else:
                return list(self.lastcenter())
        if self.is_being_tracked:
            return self.kalman.predict()[:2][:, 0]

    def addcenter(self, cen):
        self.previous_centers.append((cen[0], cen[1]))
        self.updateKF()
        self.num_frames_detected += 1
        self.num_not_found = 0
        if self.num_frames_detected >= 3:
            self.is_being_tracked = True
        if FIND_OSCILLATING:
            self.determine_oscillation(
                fps=FPS, f_0=F_0, min_frames=100)  # CHANGE 1000 TO 100

    def drop(self):
        self.num_not_found += 1
        if self.num_not_found > MAX_KALMAN_LEARNING_TIME:
            self.is_being_tracked = False

    def track_points(self):
        if self.is_being_tracked:
            return (self.previous_centers[-2], self.previous_centers[-1])

    def get_mean_drift(self, min_frames=100):
        """
        min_frames: the minimum number of frames the objct must be tracked in to be considered in the calculation

        """
        if self.num_frames_detected >= min_frames:
            initial_center = self.firstcenter()
            final_center = self.lastcenter()
            this_x_drift = (
                final_center[0] - initial_center[0]) / float(self.num_frames_detected)
            this_y_drift = (
                final_center[1] - initial_center[1]) / float(self.num_frames_detected)
            self.mean_x_drift = this_x_drift
            self.mean_y_drift = this_y_drift
        else:
            self.mean_x_drift = None
            self.mean_y_drift = None

    def determine_oscillation(self, fps=FPS, f_0=F_0, min_frames=100):
        """
        fps: sampling frequency of motion i.e. # of frames per second recorded
        f_0: the frequency we are investigating oscillation at
        min_frames: the minimum number of frames the objct must be tracked in to be considered in the calculation
        """
        if fps < 2 * f_0:
            raise ValueError(
                'sampling frequency does not satisfy Nyquist sampling theorem!')

        if self.num_frames_detected < min_frames:
            self.fft_frequencies = None
            self.x_fft = None
            self.y_fft = None
            self.is_oscillating = False
            return

        initial_center = self.firstcenter()
        x_pos = np.array([c[0] - initial_center[0]
                          for c in self.previous_centers])
        y_pos = np.array([c[1] - initial_center[1]
                          for c in self.previous_centers])

        n = len(self.previous_centers)
        len_out = n // 2 + 1
        maxf = fps / 2.0 if n % 2 == 0 else fps * (n - 1) / (2.0 * n)
        self.fft_frequencies = np.log10(
            maxf * np.arange(1, len_out) / len_out).astype(np.float32)
        f_0_index = np.argmin(np.abs(self.fft_frequencies - np.log10(f_0)))
        x_fft = np.fft.rfft(np.array(x_pos))
        y_fft = np.fft.rfft(np.array(y_pos))

        x_amp = np.abs(x_fft).astype(np.float32)
        self.x_fft = np.log10(x_amp)[1:] / np.log10(x_amp.max())
        y_amp = np.abs(y_fft).astype(np.float32)
        self.y_fft = np.log10(y_amp)[1:] / np.log10(y_amp.max())

        _iter = 20
        _threshold = 0.2
        good_frac = 0.5
        x_osc = lfit_cython.linear_ransac1D(
            self.fft_frequencies, self.x_fft, _iter, _threshold, good_frac, f_0_index)
        y_osc = lfit_cython.linear_ransac1D(
            self.fft_frequencies, self.y_fft, _iter, _threshold, good_frac, f_0_index)
        self.is_oscillating = x_osc or y_osc

    def show_fft(self, p, axis, color='red', display=False):
        if axis == 'x':
            p.line(self.fft_frequencies, self.x_fft, color=color)
        elif axis == 'y':
            p.line(self.fft_frequencies, self.y_fft, color=color)
        if display:
            show(p)


class Waitbar(object):

    def __init__(self, winname, size=[500, 100], color=[0, 0, 255],txtsize=0.5):
        self.winname = winname
        self.color = np.array(color)
        self.window = cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        self.winsize = size
        cv2.resizeWindow(self.winname, size[0], size[1])
        self.blank = 255 * np.ones((size[1], size[0], 3), dtype=np.uint8)
        self.pixel_level = 0
        self.start_time = time.time()
        self.txtsize=txtsize

    def update(self, level):
        remaining = self.estimate_time_remaining(level)
        image = np.copy(self.blank)
        self.pixel_level = int(level * self.winsize[0])

        image[int(0.3 * self.winsize[1]):-int(0.3 * self.winsize[1]),
              :self.pixel_level, :] = self.color
        msg = '{:.2f} % Done'.format(level * 100)
        cv2.putText(image, msg, (0, int(0.2 * self.winsize[1])),
                    cv2.FONT_HERSHEY_COMPLEX, self.txtsize, (0, 0, 0))
        sec = int(remaining - 60 * (remaining // 60))
        msg = 'Time remaining: {} min, {} seconds'.format(
            int(remaining // 60), sec)
        cv2.putText(image, msg, (0, int(0.9 * self.winsize[1])),
                    cv2.FONT_HERSHEY_COMPLEX, self.txtsize, (0, 0, 0))
        return image

    def estimate_time_remaining(self, level):
        speed = level / (time.time() - self.start_time)
        remaining = (1 / speed) - level
        return remaining


def nms(data, th=0.1, w=13):
    xs = data[0]
    ys = data[1]
    scores = data[2]
    indices = np.argsort(scores)[::-1]
    idxs = indices[:]
    picked = []
    while(len(indices) > 0):
        picked.append(indices[0])
        indices = indices[1:][~np.bitwise_and(np.abs(
            xs[indices[0]] - xs[indices[1:]]) < w, np.abs(ys[indices[0]] - ys[indices[1:]]) < w)]
    return [xs[picked], ys[picked]]


def computepairwise(matrix1, matrix2):
    assert len(matrix1.shape) == 2, 'First argument is not 2D'
    assert len(matrix2.shape) == 2, 'Second argument is not 2D'
    assert matrix1.shape[1] == matrix2.shape[
        1], 'Matrices have different number of features'

    result = np.zeros((matrix1.shape[0], matrix2.shape[0]), dtype=np.float32)
    for feature in range(matrix1.shape[1]):
        diff = (np.repeat(matrix1[:, feature][:, None], matrix2.shape[
                0], axis=1) - matrix2[:, feature][:, None].T)  # ,axis=1
        # print(diff.shape,matrix1.shape[0],matrix2.shape[0])
        assert diff.shape == (matrix1.shape[0], matrix2.shape[
                              0]), 'there is a bug in your program'
        result += diff**2
    return np.sqrt(result)


def matchcentertoobj(centers, tracked_objs, frame_idx):
    current_predictions = np.array(
        [list(obj.lastcenter()) for obj in tracked_objs])  # list(obj.lastcenter())
    # current_predictions=current_predictions[:,:,0] #obj.predictnow()
    # print(current_predictions.shape)
    # Nx2 array
    # centers is Mx2 array
    # compute pairwise distances (NxM)
    # if M<N be careful
    # if M >= N, possibly match existing centers to new centers if distance is below a threshold,
    # maintain a list of used indices
    # match existing centers to that new center with which it has minimum
    # distance
    centers = np.array(centers)
    # print(current_predictions.shape)
    distance = computepairwise(current_predictions, centers)  # NxM
    # print(distance)
    possible_matches = np.argmin(distance, axis=1)
    used_indices = []

    for idx, match in enumerate(possible_matches):
        # if match occurs more than once, choose the minimum distance
        candidates = []
        candidates.append(distance[idx, match])
        for idx2 in range(len(possible_matches[idx + 1:])):
            if match == possible_matches[idx + 1 + idx2]:
                candidates.append(distance[idx + 1 + idx2, match])

        # if len(candidates)>1:
        #    pass
            # print('Duplicate matches found') #this happens VERY often
        if np.argmin(candidates) != 0:
            # this means another point has lower distance than this point, so
            # this point has no matches
            tracked_objs[idx].drop()
        else:
            # print(candidates)
            if candidates[0] < 50:
                if possible_matches[idx] not in used_indices:
                    tracked_objs[idx].addcenter(centers[possible_matches[idx]])
                    tracked_objs[idx].tracked_frame_indices.append(frame_idx)
                    used_indices.append(possible_matches[idx])
            else:
                tracked_objs[idx].drop()


def draw_full_paths_of_these_beads(initial_frame, beads_ids, tracked_objs, color='green'):
    '''
    initial_frame: A clean frame on which paths are to be drawn
    bead_nos: a list containing ids of beads to draw
    '''
    written_frame = initial_frame[:]
    blank = np.zeros(
        (initial_frame.shape[0], initial_frame.shape[1]), dtype=np.uint8)
    for idx in beads_ids:
        obj = tracked_objs[idx]
        for cidx in range(1, len(obj.previous_centers)):
            blank = cv2.line(blank, obj.previous_centers[
                             cidx - 1], obj.previous_centers[cidx], 255, 1)
        textid = str(idx)
        cv2.putText(written_frame, textid, obj.lastcenter(),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))

    channels = {'blue': 0, 'green': 1, 'red': 2}
    idx = channels[color]
    data32 = initial_frame[:, :, idx].astype(np.int32)
    np.clip(data32 + blank, 0, 255, out=data32)
    written_frame[:, :, idx] = data32.astype(np.uint8)

    return written_frame


def drawtrajectory(previous, tracked_objs, this_frame, bead_indices, color='green'):
    # previous: a dark frmae like matrix with only the trajectories drawn
    # this_frame: frame on which to draw trajectory
    channels = {'blue': 0, 'green': 1, 'red': 2}

    for _beadidx in bead_indices:
        if tracked_objs[_beadidx].is_being_tracked:
            previous = cv2.line(previous, tracked_objs[_beadidx].track_points()[
                                0], tracked_objs[_beadidx].track_points()[1], 255, 1)

    idx = channels[color]
    #this_frame[:,:,:] = this_frame[:,:,:]*((previous[:,:])[:,:,np.newaxis])
    data32 = this_frame[:, :, idx].astype(np.int32)
    np.clip(data32 + previous, 0, 255, out=data32)
    this_frame[:, :, idx] = data32.astype(np.uint8)
    return previous, this_frame


def writedistances(frame, tracked_objs):
    finddist = lambda tp1, tp2: np.sqrt(
        (tp1[0] - tp2[0])**2 + (tp1[1] - tp2[1])**2)
    copied = frame[:]
    for idx, obj in enumerate(tracked_objs):
        if True:  # obj.num_frames_detected > 5:
            center = lambda: tuple(
                (np.array(obj.previous_centers[0]) + np.array(obj.previous_centers[-1])) // 2)
            textid = str(idx)
            cv2.putText(copied, textid, obj.lastcenter(),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255))
    return copied


def get_mean_drift(objs, min_frames=100):
    """
    objs: tracked_objs, a list of beads (MovingObj) being tracked
    min_frames: the minimum number of frames an objct must be tracked in to be considered in the calculation

    """
    x_drift = 0.0
    y_drift = 0.0
    num_beads_counted = 0
    for obj in objs:
        if obj.num_frames_detected >= min_frames:
            num_beads_counted += 1
            initial_center = obj.previous_centers[0]
            final_center = obj.previous_centers[-1]
            this_x_drift = (
                final_center[0] - initial_center[0]) / float(obj.num_frames_detected)
            this_y_drift = (
                final_center[1] - initial_center[1]) / float(obj.num_frames_detected)
            x_drift += this_x_drift
            y_drift += this_y_drift


def save_beads(filename, tracked_objs):
    with open(filename, 'w') as f:
        pos_dict = {idx: obj.previous_centers for idx,
                    obj in enumerate(tracked_objs)}
        time_dict = {idx: obj.tracked_frame_indices for idx,
                     obj in enumerate(tracked_objs)}
        combined = [pos_dict, time_dict]
        f.write(str(combined))


def load_beads(filename):
    loaded_beads = []
    with open(filename, 'r') as f:
        beads_dict = eval(f.read())[0]
    for bead_num in sorted(beads_dict.keys()):
        _bead = MovingObj((0, 0))
        _bead.previous_centers = beads_dict[bead_num]
        _bead.num_frames_detected = len(_bead.previous_centers)
        loaded_beads.append(_bead)
    return loaded_beads


def text2csv(fname):
    with open(fname, 'r') as f:
        bead_positions = eval(f.read())[0]

    f = open(fname[:fname.rfind('.')] + '.csv', 'w')
    writer = csv.writer(f)
    bead_numbers = sorted(list(bead_positions.keys()), key=lambda x: len(
        bead_positions[x]), reverse=True)
    duplicated = []
    for b in bead_numbers:
        duplicated.extend([str(b) + '-X', str(b) + '-Y'])

    writer.writerow(duplicated)
    max_idx = len(bead_positions[bead_numbers[0]])
    for idx in range(max_idx):
        beads_in_this_row = len(
            [b for b in bead_numbers if len(bead_positions[b]) > idx])
        row = []
        for b in bead_numbers[:beads_in_this_row]:
            row.extend(list(bead_positions[b][idx]))
        writer.writerow(row)

    f.close()


def highlight_stopped_beads(frame, tracked_objs, total_frames, bead_radius, std_threshold=1.0, strict=True, end=-1):
    n_stopped = 0
    stopped_idxs = []
    for idx, obj in enumerate(tracked_objs):
        if len(obj.previous_centers) < 2:
            is_stopped = True
        elif len(obj.previous_centers) >= 0.5 * total_frames:
            cen_x, cen_y = list(zip(*obj.previous_centers[end - 100:end]))
            cx, cy = np.std(cen_x) <= std_threshold, np.std(
                cen_y) <= std_threshold
            # conditions for satisfying stopping criteria

            is_stopped = (cx and cy) if strict else (cx or cy)
        else:
            is_stopped = False

        if is_stopped:
            n_stopped += 1
            stopped_idxs.append(idx)
            frame = cv2.circle(
                frame, obj.previous_centers[-1], bead_radius, (0, 0, 255), -1)
    print(('Number of stopped beads={}'.format(n_stopped)))
    return frame, n_stopped, stopped_idxs


def save_to_audio(tracked_objs, obj_nums, folder):
    for num in obj_nums:
        bx, by = list(zip(*tracked_objs[num].previous_centers))
        bx, by = np.array(bx), np.array(by)
        bx -= bx[0]
        by -= by[0]
        #video_time_steps = np.arange(len(bx)) / float(FPS)
        p = figure()
        p.line(np.arange(len(bx)) / float(FPS), bx,
               color='red', name='{}_x'.format(num))
        p.line(np.arange(len(by)) / float(FPS), by,
               color='blue', name='{}_y'.format(num))
        export_png(p, folder + '{}_bead.png'.format(num))

        audio_combined = compute_audio_data(bx, by)
        # print(audio_combined.shape)
        #print('Bead {}: correct_samples={},returned_samples={}'.format(num,AUDIO_RATE*bx.size/float(FPS),audio_combined.shape[0]))
        print(('Bead {}: correct time={}s'.format(num, bx.size / float(FPS))))
        wavfile.write(folder + 'bead_{}.wav'.format(num),
                      AUDIO_RATE, audio_combined)


def compute_audio_data(bx, by):
    n_seconds = len(bx) / float(FPS)
    stretch_factor = 1500
    video_time = np.arange(len(bx)) / float(FPS)
    x_i = interp1d(video_time, bx, kind='nearest')
    y_i = interp1d(video_time, by, kind='nearest')

    stretched_time = np.linspace(0, n_seconds, n_seconds * AUDIO_RATE)
    stretched_time = stretched_time[stretched_time <= video_time.max()]

    audio_x = x_i(stretched_time)
    audio_y = y_i(stretched_time)

    scale2audio = lambda x: 65535 * \
        (x - x.min()) / float(x.max() - x.min()) - 32768
    audio_combined = np.concatenate(
        (scale2audio(audio_x)[:, None], scale2audio(audio_y)[:, None]), axis=1)
    return audio_combined


def compute_audio_data2(bx, by):
    n_seconds = len(bx) / float(FPS)
    stretch_factor = 1500
    x_fft = np.fft.fft(bx)
    y_fft = np.fft.fft(by)
    true_frequencies = np.fft.fftfreq(bx.size, 1.0 / float(FPS))

    fx_r = interp1d(true_frequencies, x_fft.real, kind='nearest')
    fx_i = interp1d(true_frequencies, x_fft.imag, kind='nearest')
    fy_r = interp1d(true_frequencies, y_fft.real, kind='nearest')
    fy_i = interp1d(true_frequencies, y_fft.imag, kind='nearest')

    stretched_frequencies = np.linspace(
        0, true_frequencies.max(), (n_seconds * AUDIO_RATE // 2))
    stretched_frequencies = stretched_frequencies[
        stretched_frequencies < true_frequencies.max()]  # filter out the edges of bins

    single2doublesidedfft = lambda x: np.concatenate((x[1:][::-1], x))
    interpx_r = fx_r(stretched_frequencies)
    interpx_i = fx_i(stretched_frequencies)
    interpy_r = fy_r(stretched_frequencies)
    interpy_i = fy_i(stretched_frequencies)

    stretched_x_fft = np.complex128(np.zeros_like(interpx_r))
    stretched_y_fft = np.complex128(np.zeros_like(interpy_r))

    stretched_x_fft.real = interpx_r
    stretched_x_fft.imag = interpx_i
    stretched_y_fft.real = interpy_r
    stretched_y_fft.imag = interpy_i

    # print(stretched_x_fft.shape,stretched_y_fft.shape)

    # stretched_x_fft=single2doublesidedfft(stretched_x_fft)
    # stretched_y_fft=single2doublesidedfft(stretched_y_fft)

    stretched_x_time = np.abs(np.fft.ifft(stretched_x_fft))[:, None]
    stretched_y_time = np.abs(np.fft.ifft(stretched_y_fft))[:, None]

    audio_x = 65535 * (stretched_x_time - stretched_x_time.min()) / \
        (stretched_x_time.max() - stretched_x_time.min()) - 32768
    audio_y = 65535 * (stretched_y_time - stretched_y_time.min()) / \
        (stretched_y_time.max() - stretched_y_time.min()) - 32768

    audio_combined = np.concatenate((audio_x, audio_y), axis=1)

    return audio_combined


def get_last_frame(fname):
    video = cv2.VideoCapture(fname)
    #video.set(cv2.CAP_PROP_POS_AVI_RATIO, 0.99)
    number_of_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    # print(number_of_frames)
    video.set(cv2.CAP_PROP_POS_FRAMES, number_of_frames - 2)
    ret, frame = video.read()
    last_frame = frame[:]
    video.release()
    return last_frame


def trim_video(source, outfile, start, end):
    #source.set(cv2.CAP_PROP_POS_FRAMES, 0)  # start at the beginning
    fps = source.get(cv2.CAP_PROP_FPS)

    size = (int(source.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(source.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(fps, size)
    if os.path.exists(outfile):
        os.remove(outfile)
    sink = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(
        *FOURCC), fps, size)
    source.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    n_frames_needed = int((end - start) * fps)
    ret, frame = source.read()
    count = 1
    while count < n_frames_needed:
        sink.write(frame)
        ret, frame = source.read()
        if not ret:
            print('Reached end of file')
            break
        count += 1
    print("Finished trimming {}".format(outfile))
    sink.release()


def extract_videos_for_processing(target_folder, extract_template=False, filemode=False, guivar=None):
    all_outfiles = []
    if filemode:
        target_files = [target_folder[target_folder.rfind('/') + 1:]]
        target_folder = target_folder[:target_folder.rfind('/') + 1]
        analysis_folder = target_folder[
            :target_folder.rfind('/') + 1] + 'tracking/'

    else:
        target_files = [f for f in os.listdir(
            target_folder) if f.endswith('.mov')]
        analysis_folder = target_folder + 'tracking/'

    if not os.path.isdir(analysis_folder):
        os.mkdir(analysis_folder)

    for idx, srcfile in enumerate(target_files):
        analysis_subfolder = analysis_folder + \
            srcfile[:srcfile.rfind('.')] + '/'
        infile = target_folder + srcfile
        print(infile)
        source = cv2.VideoCapture(infile)
        n_clips = 1 + int(source.get(cv2.CAP_PROP_FRAME_COUNT) /
                          (60 * source.get(cv2.CAP_PROP_FPS)))

        if not os.path.isdir(analysis_subfolder):
            os.mkdir(analysis_subfolder)
            for min_idx in range(1, n_clips):
                if guivar:
                    guivar[0].set('Processing Video {}/{}, Trimming clip {}/{}'.format(
                        idx + 1, len(target_files), min_idx, n_clips - 1))
                    guivar[1].update_idletasks()
                time_folder = analysis_subfolder + '{}m/'.format(min_idx)
                os.mkdir(time_folder)
                outfile = time_folder + \
                    srcfile[:srcfile.rfind('.')] + '_{}m.mov'.format(min_idx)
                trim_video(source, outfile, min_idx * 60 - 10, min_idx * 60)
                all_outfiles.append(outfile)

                if extract_template:
                    extract_template_frames(outfile)
        source.release()

    return all_outfiles


def extract_template_frames(filename, name='temp1.jpg'):
    src = cv2.VideoCapture(filename)
    n_frames = src.get(cv2.CAP_PROP_FRAME_COUNT)
    src.set(cv2.CAP_PROP_POS_FRAMES, int(n_frames // 2))
    ret, frame = src.read()
    if ret:
        frame_name = filename[:filename.rfind('/') + 1] + name
        cv2.imwrite(frame_name, frame)
    else:
        print(('Could not read frame for file {}'.format(filename)))
    src.release()


def extract_temp_from_folder(target_folder):
    target_files = [f for f in os.listdir(target_folder) if f.endswith('.mov')]
    for file in target_files:
        imgname = file[:file.rfind('.')] + '.jpg'
        extract_template_frames(target_folder + file, name=imgname)


def find_min_dist(bounds, gray_binary, line_length, x_center, y_center, _theta, sign=1):
    for r in range(line_length):
        pointx = int(x_center + sign * r * np.cos(_theta))
        pointy = int(y_center + sign * r * np.sin(_theta))
        if bounds(pointx, pointy):
            if gray_binary[pointx, pointy]:
                min_dist_found = r
                return min_dist_found


def find_boundaries(imgname, debug=False):
    image = cv2.imread(imgname, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (5, 5))
    _max = gray.max()
    th, gray = cv2.threshold(gray, 0.9 * _max, _max, cv2.THRESH_BINARY)
    gray_binary = (gray > 0)
    x, y = np.where(gray > 1)
    y_center, x_center = int(y.mean()), int(x.mean())
    edges = cv2.Canny(gray, 100, 200)
    cv2.imwrite(imgname[:imgname.rfind('/') + 1] + 'check1.jpg', gray)

    line_length = 1200
    theta_amp = 24 * np.pi / 180
    theta_list = []
    rho_list = []
    bounds = lambda px, py: px < image.shape[
        0] and px >= 0 and py < image.shape[1] and py >= 0
    endpoint = lambda d, t: (
        int(y_center + d * np.sin(t)), int(x_center + d * np.cos(t)))

    for idx in range(200):
        _theta = theta_amp * (-1 + idx / 100.0)
        r = find_min_dist(bounds, gray_binary, line_length,
                          x_center, y_center, _theta, sign=1)
        theta_list.append(_theta)
        rho_list.append(r)

    if debug:
        plt.plot(theta_list, rho_list, 'r')
        plt.show()
    tilt_angle = theta_list[np.argmin(rho_list)]
    print(('Pattern is titled by {:.2f} degree'.format(
        tilt_angle * 180 / np.pi)))

    min_dist_py = np.nanmin(np.array(rho_list, dtype=np.int32))
    # print(min_dist_py)
    min_dist_my = find_min_dist(
        bounds, gray_binary, line_length, x_center, y_center, tilt_angle, sign=-1)
    min_dist_px = find_min_dist(
        bounds, gray_binary, line_length, x_center, y_center, tilt_angle + np.pi / 2, sign=1)
    min_dist_mx = find_min_dist(
        bounds, gray_binary, line_length, x_center, y_center, tilt_angle + np.pi / 2, sign=-1)

    pointxmin = endpoint(-min_dist_mx, np.pi / 2 + tilt_angle)
    pointxmax = endpoint(min_dist_px, np.pi / 2 + tilt_angle)
    pointymin = endpoint(-min_dist_my, tilt_angle)
    pointymax = endpoint(min_dist_py, tilt_angle)

    midx = ((pointxmin[0] + pointxmax[0]) // 2)
    midy = ((pointymin[1] + pointymax[1]) // 2)

    cv2.line(image, (y_center, x_center), pointymax, 255, 2)
    cv2.line(image, (y_center, x_center), pointymin, (0, 255, 0), 2)
    cv2.line(image, (y_center, x_center), pointxmax, (0, 0, 255), 2)
    cv2.line(image, (y_center, x_center), pointxmin, (255, 255, 255), 2)

    cv2.circle(image, (midx, midy), (min_dist_py + min_dist_my) // 2, 255, 2)

    ylim = lambda y0: (pointxmin[0] + (y0 - pointxmin[1]) / np.tan(tilt_angle + np.pi / 2),
                       (pointxmax[0] + (y0 - pointxmax[1]) / np.tan(tilt_angle + np.pi / 2)))
    xlim = lambda x0: (pointymin[1] + np.tan(tilt_angle) * (x0 - pointymin[0]),
                       pointymax[1] + np.tan(tilt_angle) * (x0 - pointymax[0]))

    is_in_square = lambda x0, y0: x0 < xlim(y0)[1] and x0 > xlim(y0)[0] \
        and y0 < ylim(x0)[1] and y0 > ylim(x0)[0]

    for idx in range(1000):
        pt = (int(3840 * np.random.random()), int(2160 * np.random.random()))
        if is_in_square(pt[1], pt[0]):
            cv2.circle(image, pt, 6, (0, 255, 0), -1)
        else:
            cv2.circle(image, pt, 6, (0, 0, 255), -1)

    cv2.imwrite(imgname[:imgname.rfind('/') + 1] + 'check2.jpg', image)

    return xlim, ylim, is_in_square


def find_beads_in_sensing_area(fname, tracked_objs, total_frames, bead_radius, strict=True, debug=False, oldres=None):
    frame = get_last_frame(fname) if fname.endswith(
        '.mov') else cv2.imread(fname, 1)
    outname = fname[:fname.rfind('/') + 1] + 'last_frame.jpg'
    cv2.imwrite(outname, frame)

    try:
        xlim, ylim, is_in_square = find_boundaries(outname, debug=debug)
    except Exception as e:
        print(('Error in finding beads. ' + str(e)))
        xlim, ylim, is_in_square = oldres  # works only if the first one doesn't work
        print('Successfully recovered parameters from previous result')

    beads_in_sensing_area = []
    for t in tracked_objs:
        if is_in_square(t.previous_centers[-1][1], t.previous_centers[-1][0]):
            beads_in_sensing_area.append(t)

    frame, n_stopped, _ = highlight_stopped_beads(
        frame, beads_in_sensing_area, total_frames, bead_radius, std_threshold=1.0, strict=strict, end=-1)
    return (frame, n_stopped, len(beads_in_sensing_area), (xlim, ylim, is_in_square))


def plot_pos_freq(tracked_objs, bnums, htmlname, fs=24.0, coord='x', pinam=6 / 1.0):
    pixels_in_a_micron = pinam
    figs = []
    p1 = figure()
    p2 = figure(x_axis_type="log")  # ,y_axis_type="log")
    p3 = figure(x_axis_type="log")  # ,y_axis_type="log")
    colors = ['red', 'green', 'blue', 'black', 'orange',
              'firebrick', 'fuchsia', 'indigo', 'magenta']
    for b_num in bnums:
        if coord == 'x':
            pos = [
                c[0] / pixels_in_a_micron for c in tracked_objs[b_num].previous_centers]
        elif coord == 'y':
            pos = [
                c[1] / pixels_in_a_micron for c in tracked_objs[b_num].previous_centers]

        #l2dist=lambda tuple1,tuple2: np.sqrt((tuple1[0]-tuple2[0])**2+(tuple1[1]-tuple2[1])**2)/6.0
        pos = [posn - pos[0] for posn in pos]
        p1.line([idx / float(fs) for idx in range(len(pos))], pos,
                legend='Position (#' + str(b_num) + ')', color=colors[bnums.index(b_num)])
        n = len(pos)
        len_out = n // 2 + 1
        maxf = fs / 2.0 if n % 2 == 0 else fs * (n - 1) / (2.0 * n)
        frequencies = maxf * np.arange(len_out) / len_out
        fftarr = np.fft.rfft(np.array(pos))
        ys = np.abs(fftarr)
        # print(fftarr)
        phase = np.arctan(np.imag(fftarr) / np.real(fftarr))
        # print(phase)
        #ys[1:]*=2 #
        p2.line(frequencies, np.log10(ys) / np.max(np.log10(ys)),
                legend='Frequency (#' + str(b_num) + ')', color=colors[bnums.index(b_num)])
        p3.line(frequencies, phase, legend='Phase (#' +
                str(b_num) + ')', color=colors[bnums.index(b_num)])

    p1.xaxis.axis_label = 'Time (s)'
    p1.yaxis.axis_label = 'Position (um)'
    p1.legend.click_policy = 'hide'
    p1.legend.location = 'top_right'
    p2.xaxis.axis_label = 'Frequency (Hz)'
    p2.yaxis.axis_label = 'Normalized log(Amplitude)'
    p2.legend.click_policy = 'hide'
    p3.xaxis.axis_label = 'Frequency (Hz)'
    p3.yaxis.axis_label = 'Phase'
    p3.legend.click_policy = 'hide'
    figs.append([p1, p2])
    figs.append([p3])
    output_file(htmlname, title='Analysis of beads position')
    grid = gridplot(figs)
    show(grid)


def count_beads(img, tmp, threshold=0.8, bead_radius=3):
    frame = cv2.imread(img, 1)
    frame_orig = frame[:]
    gray = cv2.imread(img, 0)
    template = cv2.imread(tmp, 0)
    print((frame.shape))
    print((template.shape))
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    loc = [np.where(res >= threshold)[0], np.where(
        res >= threshold)[1], res[np.where(res >= threshold)]]
    loc = nms(loc)
    tracked_objs = []
    txtfile = img[:img.rfind('.')] + '_data.txt'
    for pt in zip(*loc[::-1]):
        center = (pt[0] + bead_radius // 2, pt[1] + bead_radius // 2)
        frame = cv2.circle(frame, center, bead_radius, (0, 255, 0), 1)
        tracked_objs.append(MovingObj(center))
    highlighted, num_stopped, stopped_idxs = highlight_stopped_beads(
        frame_orig, tracked_objs, 1, bead_radius, std_threshold=1.0)
    cv2.imwrite(img[:img.rfind('.') + 1] +
                '_{}_stopped_beads.jpg'.format(num_stopped), highlighted)
    with open(img[:img.rfind('/') + 1] + 'num_tracked.txt', 'w') as f:
        f.write('Number of beads tracked={}\n Number of beads stopped= {}\n Percentage of beads stopped= {:.2f}'
                .format(len(tracked_objs), num_stopped, num_stopped * 100.0 / float(len(tracked_objs))))
    save_beads(txtfile, tracked_objs)


def analyze_pictures(folder):
    filenames = []
    subfolders = sorted(
        [folder + f for f in os.listdir(folder) if os.path.isdir(folder + f)])
    for f in subfolders:
        for g in sorted(os.listdir(f)):
            if g.endswith('.jpg') and 'temp1' not in g:
                imagename = f + '/' + g
                print(imagename)
                template_file = f + '/temp1.jpg'
                count_beads(imagename, template_file)
                filenames.append(imagename)
    return filenames


def extract_frames(fname, frame_numbers):
    source = cv2.VideoCapture(fname)
    folder = fname[:fname.rfind('.mov')]
    if not os.path.exists(folder):
        os.mkdir(folder)

    for idx, f in enumerate(frame_numbers):
        source.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = source.read()
        if ret:
            cv2.imwrite(folder + '/frame{}.jpg'.format(idx), frame)
        else:
            print(('Could not read frame number {}'.format(f)))

    source.release()


def controlled_play(fname):
    src = cv2.VideoCapture(fname)
    ret, frame = src.read()
    cv2.namedWindow("Video Player", cv2.WINDOW_NORMAL)
    nframes = int(src.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = src.get(cv2.CAP_PROP_FPS)
    onTrackbarSlide = lambda n: src.set(cv2.CAP_PROP_POS_FRAMES, n)
    cv2.createTrackbar("Position", "Video Player", 0, nframes, onTrackbarSlide)
    prev_pos = 0
    ret, frame = src.read()
    play = True

    while True:
        # int(src.get(cv2.CAP_PROP_POS_FRAMES))
        current_pos = cv2.getTrackbarPos('Position', 'Video Player')
        cv2.resizeWindow("Video Player", 960, 540)

        if current_pos != prev_pos:
            play = not play
            src.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
            ret, frame = src.read()
            prev_pos = current_pos

        elif play:
            ret, frame = src.read()
            current_pos = int(src.get(cv2.CAP_PROP_POS_FRAMES))
            prev_pos = current_pos

        cv2.imshow(fname[fname.rfind('/') + 1:], frame)
        cv2.setTrackbarPos("Position", "Video Player", current_pos)
        # print(current_pos)
        k = cv2.waitKey(1)
        if k == 27:
            break
        elif k == 32:  # space
            play = not play

        elif k == 115:  # s
            cv2.imwrite(fname[:fname.rfind('/') + 1] + 'frame.jpg', frame)
            print('Current frame saved')

    cv2.destroyAllWindows()
    src.release()


def crop_and_trim(fname, prev_points=None):
    src = cv2.VideoCapture(fname)
    ret, frame = src.read()
    winname = fname[fname.rfind('/') + 1:]
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname, cropwindow[0], cropwindow[1])
    nframes = int(src.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = src.get(cv2.CAP_PROP_FPS)
    onTrackbarSlide = lambda n: src.set(cv2.CAP_PROP_POS_FRAMES, n)
    cv2.createTrackbar("Position", winname, 0, nframes, onTrackbarSlide)
    prev_pos = 0
    ret, frame = src.read()
    points_list = []

    get_points = lambda e, x, y, flags, param: points_list.append(
        (x, y)) if e == cv2.EVENT_LBUTTONDOWN else None
    cv2.setMouseCallback(winname, get_points)
    play = False

    while True:
        current_pos = cv2.getTrackbarPos('Position', winname)

        if current_pos != prev_pos:
            play = not play
            src.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
            ret, frame = src.read()
            prev_pos = current_pos

        elif play:
            ret, frame = src.read()
            current_pos = int(src.get(cv2.CAP_PROP_POS_FRAMES))
            prev_pos = current_pos

        newframe=cv2.resize(frame,cropwindow)
        yr,xr=(frame.shape[0]//cropwindow[1],frame.shape[1]//cropwindow[0])
        cv2.imshow(winname, newframe)
        cv2.setTrackbarPos("Position", winname, current_pos)
        # print(current_pos)
        k = cv2.waitKey(30)
        if k == 27:
            break
        elif k == ord('p'):
            if prev_points:
                points_list = prev_points[:]
                print('Using previous dimensions')
                break
            else:
                print('No previous points specified')

    x1, y1 = points_list[0]
    x2, y2 = points_list[-1]

    assert x2 >= x1, 'x2<x1'
    assert y2 >= y1, 'y2<y1'

    print(points_list)
    src.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not os.path.exists(fname[:fname.rfind('/') + 1] + 'tracking/'):
        os.mkdir(fname[:fname.rfind('/') + 1] + 'tracking/')

    newname = fname[:fname.rfind('/') + 1] + \
        'tracking/{}'.format(fname[fname.rfind('/') + 1:])
    size = (int(xr*(x2 - x1)), int(yr*(y2 - y1)))
    print(size)

    if os.path.exists(newname):
        os.remove(newname)

    sink = cv2.VideoWriter(newname, cv2.VideoWriter_fourcc(
        *FOURCC), fps, size)
    ret, frame = src.read()

    while ret:
        subframe = frame[int(yr*y1):int(yr*y2), int(xr*x1):int(xr*x2), :]
        sink.write(subframe)
        ret, frame = src.read()

    cv2.destroyAllWindows()
    src.release()
    sink.release()
    cv2.destroyAllWindows()
    print('Source and sink released')
    return newname, points_list


def filename_chunks(full_list):
    '''function which splits file names into many chunks'''
    chunks=[]
    video_names=[f[:f[:f.rfind('/')].rfind('/')+1] for f in full_list]
    #[f.split('/') for f in full_list]
    counts=Counter(video_names)
    for k,v in counts.items():
        c=[]
        for idx in range(1,v+1):
            vidname=[f for f in os.listdir(k+'{}m/'.format(idx)) if f.endswith('.mov') and 'analyzed_' not in f][0]
            c.append(k+'{}m/'.format(idx)+vidname)
        chunks.append(c)
    return chunks

def analyze_sensing_area(files_to_analyze,bead_radius=3,total_frames=240,debug=False):
    oldres=None
    failed=[]
    succeeded=[]
    maxtries=2*len(files_to_analyze)
    ntries=0
    for fname in files_to_analyze:
        txtfile=fname[:fname.rfind('.')]+'_data.txt'
        tracked_objs=[]
        tracked_objs=load_beads(txtfile)
        ntries+=1
        try:
            highlighted_sa,num_stopped_sa,num_in_sa,oldres=find_beads_in_sensing_area(fname,tracked_objs,total_frames, bead_radius,strict=True,debug=debug,oldres=oldres)
            cv2.imwrite(fname[:fname.rfind('.')+1]+'_{}_stopped_beads_sa.jpg'.format(num_stopped_sa),highlighted_sa)
            stoppedtxt_sa=txtfile[:txtfile.rfind('.')]+'_stopped_sa.txt'
            print('Total beads in sensing area= {}'.format(num_in_sa))
            with open(stoppedtxt_sa,'w') as f:
                msg='Number of beads in sensing area={}\n Number of beads stopped={}\n Percentage of beads stopped= {:.3f}\n'.\
                format(num_in_sa,num_stopped_sa,num_stopped_sa*100.0/num_in_sa)
                f.write(msg)
            succeeded.append(fname)
        except Exception as e:
            print('Error while analyzing file: {}'.format(fname))
            print(str(e))
            failed.append(fname)

        if len(succeeded)>0 and len(failed)>0:
            files_to_analyze.extend(failed) #modifying something while iterating over it. for shame!
            failed=[]
        if ntries>=maxtries:
            print("Reached maximum number of tries. Quitting")
            return

def track_video(fname, template_file, threshold,guiflag=True,skip=1):
    tic=time.time()
    video = cv2.VideoCapture(fname)
    txtfile = fname[:fname.rfind('.')] + '_data.txt'
    filename = fname[:fname.rfind('/') + 1] + \
        'analyzed_' + fname[fname.rfind('/') + 1:]
    num_frames_in_history = NUM_FRAMES_IN_HISTORY
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)/skip)
    if guiflag:
        bar = Waitbar(filename[filename.rfind('/') + 1:],size=[700, 200], color=[0, 0, 255],txtsize=1.0)

    if os.path.exists(filename):
        os.remove(filename)

    fps = video.get(cv2.CAP_PROP_FPS)
    if (fps > 0):
        print('Successfully reading video file')

    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(
        filename, cv2.VideoWriter_fourcc(*FOURCC), fps, size)

    tracked_objs = []
    osc_color = 'red'
    nosc_color = 'green'
    template = cv2.imread(template_file, 0)
    bead_radius = template.shape[0] // 2
    possible_new_objs = []

    prev1 = np.zeros((size[1], size[0]), dtype=np.uint8)
    prev2 = np.zeros((size[1], size[0]), dtype=np.uint8)

    dist = lambda obj, cen: np.sqrt(
        np.sum((np.array(obj.previous_centers[-1]) - np.array(cen))**2))
    ret, frame = video.read()
    count = 0

    while (ret):
        count += 1
        last_frame = frame[:]
        current_centers = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        loc = [np.where(res >= threshold)[0], np.where(
            res >= threshold)[1], res[np.where(res >= threshold)]]
        loc = nms(loc)
        if guiflag:
            img = bar.update(float(count/skip) / total_frames)
            cv2.imshow(bar.winname, img)
            k = cv2.waitKey(1)
            
        for pt in zip(*loc[::-1]):
            center = (pt[0] + bead_radius // 2, pt[1] + bead_radius // 2)
            frame = cv2.circle(frame, center, bead_radius, (0, 255, 0), 1)
            if count <= num_frames_in_history:
                for obj in tracked_objs:
                    if dist(obj, center) < 5 * bead_radius:
                        break
                else:
                    tracked_objs.append(MovingObj(center))
                    tracked_objs[-1].tracked_frame_indices.append(count)
            else:
                current_centers.append(center)

        if count > num_frames_in_history:
            matchcentertoobj(current_centers, tracked_objs, frame_idx=count)
            are_oscillating = [idx for idx, b in enumerate(
                tracked_objs) if hasattr(b, 'is_oscillating') and b.is_oscillating]
            not_oscillating = [idx for idx, b in enumerate(tracked_objs) if hasattr(
                b, 'is_oscillating') and not b.is_oscillating]
            prev1, frame = drawtrajectory(
                prev1, tracked_objs, frame, are_oscillating, osc_color)
            prev2, frame = drawtrajectory(
                prev2, tracked_objs, frame, not_oscillating, nosc_color)

        videoWriter.write(frame)

        ret, frame = video.read()
        if count>=num_frames_in_history:
            while count%skip!=0: #read more frames if necessary
                ret, frame = video.read()
                count+=1

    frame = writedistances(last_frame, tracked_objs)
    videoWriter.write(frame)
    videoWriter.release()
    video.release()
    
    stoppedtxt = txtfile[:txtfile.rfind('.')] + '_stopped.txt'
    stoppedtxt_easy = txtfile[:txtfile.rfind('.')] + '_stopped_easy.txt'
    save_beads(txtfile, tracked_objs)
    cv2.imwrite(fname[:fname.rfind('.')] + '_tracked_objs.jpg', frame)
    text2csv(txtfile)
    try:
        last_frame_of_video = get_last_frame(fname)
    except Exception as e:
        print((str(e)))
        last_frame_of_video = frame[:]

    highlighted, num_stopped, stopped_idxs = highlight_stopped_beads(
        last_frame_of_video, tracked_objs, total_frames, bead_radius, std_threshold=1.0)
    highlighted_easy, num_stopped_easy, stopped_idxs_easy = highlight_stopped_beads(
        last_frame_of_video, tracked_objs, total_frames, bead_radius, std_threshold=1.0, strict=False)
    cv2.imwrite(fname[:fname.rfind('.') + 1] +
                '_{}_stopped_beads.jpg'.format(num_stopped), highlighted)
    cv2.imwrite(fname[:fname.rfind(
        '.') + 1] + '_{}_stopped_beads_easy.jpg'.format(num_stopped_easy), highlighted_easy)

    save_beads(stoppedtxt, [tracked_objs[idx] for idx in stopped_idxs])
    save_beads(stoppedtxt_easy, [tracked_objs[idx]
                                 for idx in stopped_idxs_easy])

    if num_stopped > 0:
        text2csv(stoppedtxt)

    if num_stopped_easy > 0:
        text2csv(stoppedtxt_easy)

    with open(fname[:fname.rfind('/') + 1] + 'correspondence.txt', 'w') as f:
        f.write(str([b for b in enumerate(stopped_idxs)]))

    with open(fname[:fname.rfind('/') + 1] + 'num_tracked.txt', 'w') as f:
        f.write('Number of beads tracked={}\n Number of beads stopped= {}\n Percentage of beads stopped= {:.2f}'
                .format(len(tracked_objs), num_stopped, num_stopped * 100.0 / float(len(tracked_objs))))
    if guiflag:
        cv2.destroyWindow(bar.winname)
    toc=time.time()
    #print("Time required for tracking= {:.2f}".format(toc-tic))