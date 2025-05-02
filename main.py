#!/usr/bin/env python3

import os
import argparse
import cv2
import numpy as np


def calculate_step(num_frames, target_frames):
    """
    Calculate the step size to reduce the number of frames to target_frames.
    """
    if num_frames <= target_frames:
        return 1
    else:
        step = num_frames // target_frames
        return step


def get_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def create_photo_avg(frames, short):
    # each pixel in result image will be average of that pixel in all frames
    rAvg = None
    gAvg = None
    bAvg = None
    num_frames = len(frames)
    step = 1
    if short:
        step = calculate_step(num_frames, 120)

    for i in range(0, num_frames, step):
        R, G, B = cv2.split(frames[i].astype('float32'))

        if rAvg is None:
            rAvg = R
            gAvg = G
            bAvg = B
        else:
            rAvg = cv2.accumulate(R, rAvg)
            gAvg = cv2.accumulate(G, gAvg)
            bAvg = cv2.accumulate(B, bAvg)

    # Normalize the averages to the range [0, 255]
    rAvg = cv2.normalize(rAvg, None, 0, 255, cv2.NORM_MINMAX)
    gAvg = cv2.normalize(gAvg, None, 0, 255, cv2.NORM_MINMAX)
    bAvg = cv2.normalize(bAvg, None, 0, 255, cv2.NORM_MINMAX)


    avg = cv2.merge((rAvg, gAvg, bAvg)).astype('uint8')
    return avg


def create_photo_grad(frames, short, step_cmd):
    step = step_cmd
    if short:
        step = calculate_step(len(frames), 120)

    exposure_length = step_cmd

    if short and step_cmd > 1:
        exposure_length = 1
        print('Might expect buggy behavior with --short and --step > 1')


    for i in range(exposure_length, len(frames), step):
        R, G, B = cv2.split(frames[i].astype('float32'))
        R_prev, G_prev, B_prev = cv2.split(frames[i-exposure_length].astype('float32'))
        R_grad = cv2.absdiff(R, R_prev)
        G_grad = cv2.absdiff(G, G_prev)
        B_grad = cv2.absdiff(B, B_prev)
        if i == exposure_length:
            rGrad = R_grad
            gGrad = G_grad
            bGrad = B_grad
        else:
            rGrad = cv2.accumulate(rGrad, R_grad)
            gGrad = cv2.accumulate(gGrad, G_grad)
            bGrad = cv2.accumulate(bGrad, B_grad)
    # Normalize the gradients to the range [0, 255]
    rGrad = cv2.normalize(rGrad, None, 0, 255, cv2.NORM_MINMAX)
    gGrad = cv2.normalize(gGrad, None, 0, 255, cv2.NORM_MINMAX)
    bGrad = cv2.normalize(bGrad, None, 0, 255, cv2.NORM_MINMAX)
    grad = cv2.merge((rGrad, gGrad, bGrad)).astype('uint8')
    return grad

def create_photo_blend(frames, short, step_cmd, alpha=0.5):
    """
    Blenduje priemerovaný obraz s gradientovým pomocou váhy alpha.
    alpha=0.0 -> čisto priemerovanie
    alpha=1.0 -> čisto gradient
    """
    avg = create_photo_avg(frames, short)
    grad = create_photo_grad(frames, short, step_cmd)

    # Previesť na float, aby sme mohli spraviť vážený súčet
    avg_float = avg.astype('float32')
    grad_float = grad.astype('float32')

    # Blend
    blended = cv2.addWeighted(avg_float, 1.0 - alpha, grad_float, alpha, 0)

    return blended.astype('uint8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate long exposure from video')
    parser.add_argument('--video_path', '-v', required=True, help='Path to the video file')
    parser.add_argument('--output_path', '-o', required=True, help='Path to the output file')
    parser.add_argument('--short', '-s', action='store_true', help='Use only the first 30 frames')
    parser.add_argument('--grad', '-g', action='store_true', help='Using gradients instead of averaging')
    parser.add_argument('--step', '-st', type=int, default=1, help='Step size for frame selection, (great pairing with gradient based simulation) - buggy usage with --short')
    parser.add_argument('--blend', '-b', action='store_true', help='Blend average and gradient images')
    parser.add_argument('--alpha', '-a', type=float, default=0.5, help='Alpha value for blending (0.0 to 1.0)')

    args = parser.parse_args()

    video_path = args.video_path
    output_path = args.output_path

    print('Video path:', video_path)
    print('Output path:', output_path)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not os.path.exists(os.path.dirname(output_path)):
        raise FileNotFoundError(f"Output directory not found: {os.path.dirname(output_path)}")
    
    frames = get_frames(video_path)
    resolution = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_WIDTH), cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Number of frames: {len(frames)}")

    if args.grad:
        result_image = create_photo_grad(frames, args.short, args.step)
    elif args.blend:
        result_image = create_photo_blend(frames, args.short, args.step, args.alpha)
    else:
        result_image = create_photo_avg(frames, args.short)


    cv2.imwrite(output_path, result_image)        # For now, just print the shape of the first frame

