"""
------------------------------------------------------------------------------------------------------

                                               ,@@@@@@
                                            @@@@@@@@@@@            @@@
                                          @@@@@@@@@@@@      @@@@@@@@@@@
                                        @@@@@@@@@@@@@   @@@@@@@@@@@@@@
                                      @@@@@@@/         ,@@@@@@@@@@@@@
                                         /@@@@@@@@@@@@@@@  @@@@@@@@
                                    @@@@@@@@@@@@@@@@@@@@@@@@ @@@@@
                                @@@@@@@@                @@@@@
                              ,@@@                        @@@@&
                                             @@@@@@.       @@@@
                                   @@@     @@@@@@@@@/      @@@@@
                                   ,@@@.     @@@@@@((@     @@@@(
                                   //@@@        ,,  @@@@  @@@@@
                                   @@@(                @@@@@@@
                                   @@@  @          @@@@@@@@#
                                       @@@@@@@@@@@@@@@@@
                                      @@@@@@@@@@@@@(

Adaptive Haar Surround Feature: Summer, PallasNeko (Optimization)
Algorithm App Implementations and Tweaks By: Prohurtz

Copyright (c) 2025 EyeTrackVR <3

LICENSE: Summer Software Distribution License 1.0
------------------------------------------------------------------------------------------------------
"""

import os
from logging import FileHandler, Formatter, INFO, StreamHandler, getLogger

import cv2
import numpy as np


class AHSF:
    def __init__(self, video_src, save_logfile=False, imshow_enable=False, save_video=False):
        self.this_file_basename = os.path.basename(__file__)
        self.this_file_name = self.this_file_basename.replace(".py", "")
        self.alg_ver = "PallasNekoV3"

        self.save_logfile = save_logfile
        self.imshow_enable = imshow_enable
        self.save_video = save_video

        self.VideoCapture_SRC = video_src
        self.input_is_webcam = False
        self.benchmark_flag = True if not self.input_is_webcam and not self.imshow_enable and not self.save_video else False
        self.loop_num = 1 if self.imshow_enable or self.save_video else 10
        self.output_video_path = f"./{self.this_file_name}.mp4"
        self.logfilename = f"./{self.this_file_name}.log"
        self.print_enable = False

        self.lru_maxsize_vvs = 16
        self.lru_maxsize_vs = 64
        self.lru_maxsize_s = 128

        self.logger = getLogger(__name__)
        self.logger.setLevel(INFO)
        formatter = Formatter("%(message)s")
        handler = StreamHandler()
        handler.setLevel(INFO)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        if self.save_logfile:
            handler = FileHandler(self.logfilename, encoding="utf8", mode="w")
            handler.setLevel(INFO)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        else:
            self.save_logfile = False
        self.video_wr = cv2.VideoWriter if self.save_video else None

    def filter_light(self, img_gray, img_blur, tau):
        for i in range(img_gray.shape[1]):
            for j in range(img_gray.shape[0]):
                if img_gray[j, i] > tau:
                    img_blur[j, i] = tau
                else:
                    img_blur[j, i] = img_gray[j, i]
        return img_blur

    def get_empty_array(self, frame_shape, width_min, width_max, wh_step, xy_step, roi, ratio_outer):
        frame_int_dtype = np.intc
        np_index_dtype = (
            np.intc
        )  # memo: Better to use np.intp, but a little slower ref: https://numpy.org/doc/1.25/user/basics.indexing.html#detailed-notes

        row, col = frame_shape

        frame_int = np.empty((row + 1, col + 1), dtype=frame_int_dtype)

        w_arr = np.arange(width_min, width_max + 1, wh_step, dtype=np_index_dtype)
        h_arr = (w_arr / ratio_outer).astype(np.int16)

        # memo: It is not smart code and needs to be changed.
        y_out_n = np.hstack([np.arange(roi[1] + h, roi[3] - h, xy_step, dtype=np_index_dtype) for h in h_arr])
        x_out_n = np.hstack([np.arange(roi[0] + w, roi[2] - w, xy_step, dtype=np_index_dtype) for w in w_arr])
        y_out_h = np.hstack([np.arange(roi[1] + h, roi[3] - h, xy_step, dtype=np_index_dtype) + h for h in h_arr])
        x_out_w = np.hstack([np.arange(roi[0] + w, roi[2] - w, xy_step, dtype=np_index_dtype) + w for w in w_arr])
        out_h = y_out_h - y_out_n
        out_w = x_out_w - x_out_n

        y_in_n = np.hstack([np.arange(roi[1] + h, roi[3] - h, xy_step, dtype=np_index_dtype) + int(h / 4) for h in h_arr])
        x_in_n = np.hstack([np.arange(roi[0] + w, roi[2] - w, xy_step, dtype=np_index_dtype) + int(w / 4) for w in w_arr])
        y_in_h = np.hstack(
            [np.arange(roi[1] + h, roi[3] - h, xy_step, dtype=np_index_dtype) + int(h / 4) + int(h / 2) for h in h_arr]
        )
        x_in_w = np.hstack(
            [np.arange(roi[0] + w, roi[2] - w, xy_step, dtype=np_index_dtype) + int(w / 4) + int(w / 2) for w in w_arr]
        )
        in_h = y_in_h - y_in_n
        in_w = x_in_w - x_in_n

        # # memo: Unelegant code
        # # memo: Non-transposed version
        # wh_in_arr = np.hstack([np.full(((roi[3] - h) - (roi[1] + h) - 1) // xy_step + 1,int(h/2),dtype=np_index_dtype) for h in h_arr])[:, np.newaxis] * np.hstack([np.full(((roi[2] - w) - (roi[0] + w) - 1) // xy_step + 1,int(w/2),dtype=np_index_dtype) for w in w_arr])[np.newaxis, :]
        # wh_out_arr = np.hstack([np.full(((roi[3] - h) - (roi[1] + h) - 1) // xy_step + 1,h,dtype=np_index_dtype) for h in h_arr])[:, np.newaxis] * np.hstack([np.full(((roi[2] - w) - (roi[0] + w) - 1) // xy_step + 1,w,dtype=np_index_dtype) for w in w_arr])[np.newaxis, :]

        # memo: Unelegant code
        # memo: transposed version

        wh_in_arr = (
            np.hstack(
                [
                    np.full(
                        ((roi[2] - w) - (roi[0] + w) - 1) // xy_step + 1,
                        int(w / 2),
                        dtype=np_index_dtype,
                    )
                    for w in w_arr
                ]
            )[:, np.newaxis]
            * np.hstack(
                [
                    np.full(
                        ((roi[3] - h) - (roi[1] + h) - 1) // xy_step + 1,
                        int(h / 2),
                        dtype=np_index_dtype,
                    )
                    for h in h_arr
                ]
            )[np.newaxis, :]
        )
        wh_out_arr = (
            np.hstack(
                [
                    np.full(
                        ((roi[2] - w) - (roi[0] + w) - 1) // xy_step + 1,
                        w,
                        dtype=np_index_dtype,
                    )
                    for w in w_arr
                ]
            )[:, np.newaxis]
            * np.hstack(
                [
                    np.full(
                        ((roi[3] - h) - (roi[1] + h) - 1) // xy_step + 1,
                        h,
                        dtype=np_index_dtype,
                    )
                    for h in h_arr
                ]
            )[np.newaxis, :]
        )

        mu_outer_rect = cv2.subtract(
            wh_out_arr, wh_in_arr
        )

        wh_in_arr = 1 / wh_in_arr
        mu_outer_rect = 1 / mu_outer_rect
        mu_outer_rect2 = -1.0 * mu_outer_rect

        # 1/wh_in_arr == wh_in_arr_mul
        return (
            frame_int,
            y_out_n,
            x_out_n,
            y_out_h,
            x_out_w,
            out_h,
            out_w,
            y_in_n,
            x_in_n,
            y_in_h,
            x_in_w,
            in_h,
            in_w,
            wh_in_arr,
            wh_out_arr,
            mu_outer_rect,
            mu_outer_rect2,
        )

    def coarse_detection(self, img_gray, params):
        ratio_outer = params["ratio_outer"]
        kf = params["kf"]
        width_min = params["width_min"]
        width_max = params["width_max"]
        wh_step = params["wh_step"]
        xy_step = params["xy_step"]
        roi = params["roi"]
        init_rect_flag = params["init_rect_flag"]
        init_rect = params["init_rect"]

        imgboundary = (0, 0, img_gray.shape[1], img_gray.shape[0])
        img_blur = np.copy(img_gray)

        # Assign values to avoid unassigned errors
        pupil_rect_coarse = (10, 10, 10, 10)
        outer_rect_coarse = (5, 5, 5, 5)

        if init_rect_flag:
            init_rect_down = self.rect_scale(init_rect, params["ratio_downsample"], False)
            init_rect_down = self.intersect_rect(init_rect_down, imgboundary)
            img_blur = img_gray[
                       init_rect_down[1]: init_rect_down[1] + init_rect_down[3],
                       init_rect_down[0]: init_rect_down[0] + init_rect_down[2],
                       ]

        (
            frame_int,
            y_out_n,
            x_out_n,
            y_out_h,
            x_out_w,
            out_h,
            out_w,
            y_in_n,
            x_in_n,
            y_in_h,
            x_in_w,
            in_h,
            in_w,
            wh_in_arr,
            wh_out_arr,
            mu_outer_rect,
            mu_outer_rect2,
        ) = self.get_empty_array(img_blur.shape, width_min, width_max, wh_step, xy_step, roi, ratio_outer)
        cv2.integral(
            img_blur, sum=frame_int, sdepth=cv2.CV_32S
        )

        out_p_temp = frame_int.take(y_out_n, axis=0, mode="clip")
        out_p_temp = cv2.transpose(out_p_temp)
        out_p00 = out_p_temp.take(x_out_n, axis=0, mode="clip")
        out_p01 = out_p_temp.take(x_out_w, axis=0, mode="clip")
        out_p_temp = frame_int.take(y_out_h, axis=0, mode="clip")
        out_p_temp = cv2.transpose(out_p_temp)
        out_p11 = out_p_temp.take(x_out_w, axis=0, mode="clip")
        out_p10 = out_p_temp.take(x_out_n, axis=0, mode="clip")

        outer_sum = cv2.add(out_p00, out_p11)
        cv2.subtract(outer_sum, out_p01, dst=outer_sum)
        cv2.subtract(outer_sum, out_p10, dst=outer_sum)

        in_p_temp = frame_int.take(y_in_n, axis=0, mode="clip")
        in_p_temp = cv2.transpose(in_p_temp)
        in_p00 = in_p_temp.take(x_in_n, axis=0, mode="clip")
        in_p01 = in_p_temp.take(x_in_w, axis=0, mode="clip")
        in_p_temp = frame_int.take(y_in_h, axis=0, mode="clip")
        in_p_temp = cv2.transpose(in_p_temp)
        in_p11 = in_p_temp.take(x_in_w, axis=0, mode="clip")
        in_p10 = in_p_temp.take(x_in_n, axis=0, mode="clip")

        inner_sum = cv2.add(in_p00, in_p11)
        cv2.subtract(inner_sum, in_p01, dst=inner_sum)
        cv2.subtract(inner_sum, in_p10, dst=inner_sum)

        inner_sum_f = inner_sum.astype(np.float64)
        outer_sum_f = outer_sum.astype(np.float64)

        response_value = np.empty(outer_sum.shape, dtype=np.float64)
        inout_rect_sum = mu_outer_rect2.copy()
        inout_rect_mul = mu_outer_rect.copy()

        cv2.multiply(inner_sum_f, inout_rect_mul, inout_rect_mul)
        cv2.multiply(outer_sum_f, inout_rect_sum, inout_rect_sum)
        cv2.add(inout_rect_mul, inout_rect_sum, dst=inout_rect_sum)

        cv2.multiply(inner_sum_f, wh_in_arr, inner_sum_f, kf)
        cv2.add(inout_rect_sum, inner_sum_f, dst=response_value)

        _, _, min_loc, _ = cv2.minMaxLoc(response_value)

        rec_o = (
            x_out_n[min_loc[1]],
            y_out_n[min_loc[0]],
            out_w[min_loc[1]],
            out_h[min_loc[0]],
        )
        rec_in = (
            x_in_n[min_loc[1]],
            y_in_n[min_loc[0]],
            in_w[min_loc[1]],
            in_h[min_loc[0]],
        )
        pupil_rect_coarse = rec_in
        outer_rect_coarse = rec_o

        return pupil_rect_coarse, outer_rect_coarse

    def rect_scale(self, rect, scale, round_up=True):
        x, y, width, height = rect
        new_width = int(width * scale)
        new_height = int(height * scale)
        if round_up:
            new_width = int(np.ceil(width * scale))
            new_height = int(np.ceil(height * scale))
        new_x = x + int((width - new_width) / 2)
        new_y = y + int((height - new_height) / 2)
        return new_x, new_y, new_width, new_height

    def intersect_rect(self, rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        x = max(x1, x2)
        y = max(y1, y2)
        w = min(x1 + w1, x2 + w2) - x
        h = min(y1 + h1, y2 + h2) - y
        return x, y, w, h

    def External_Run_AHSF(self, frame_gray):
        average_color = np.mean(frame_gray)
        height, width = frame_gray.shape
        max_dimension = max(height, width)
        square_background = np.full((max_dimension, max_dimension), average_color, dtype=np.uint8)
        x_offset = (max_dimension - width) // 2
        y_offset = (max_dimension - height) // 2
        square_background[y_offset:y_offset + height, x_offset:x_offset + width] = frame_gray
        frame_gray = cv2.resize(square_background, (100, 100))
        frame_clear_resize = frame_gray.copy()

        params = {
            "ratio_downsample": 0.5,
            "use_init_rect": False,
            "mu_outer": 200,
            "mu_inner": 50,
            "ratio_outer": 1,
            "kf": 1,
            "width_min": 25,
            "width_max": 50,
            "wh_step": 1,
            "xy_step": 5,
            "roi": (0, 0, frame_gray.shape[1], frame_gray.shape[0]),
            "init_rect_flag": False,
            "init_rect": (0, 0, frame_gray.shape[1], frame_gray.shape[0]),
        }
        try:
            pupil_rect_coarse, outer_rect_coarse = self.coarse_detection(frame_gray, params)

        except TypeError:
            return frame_gray, frame_gray, 0, 0, 0

        x_center = outer_rect_coarse[0] + outer_rect_coarse[2] / 2
        y_center = outer_rect_coarse[1] + outer_rect_coarse[3] / 2
        x, y, width, height = outer_rect_coarse

        cv2.circle(frame_gray, (int(x_center), int(y_center)), 2, (255, 255, 255), -1)

        cv2.rectangle(frame_gray, (pupil_rect_coarse[0], pupil_rect_coarse[1]),
                      (pupil_rect_coarse[0] + pupil_rect_coarse[2], pupil_rect_coarse[1] + pupil_rect_coarse[3]),
                      (0, 255, 0), 2)
        cv2.rectangle(frame_gray, (outer_rect_coarse[0], outer_rect_coarse[1]),
                      (outer_rect_coarse[0] + outer_rect_coarse[2], outer_rect_coarse[1] + outer_rect_coarse[3]),
                      (255, 0, 0), 2)

        return frame_gray, frame_clear_resize, x_center, y_center, abs(width - height)
