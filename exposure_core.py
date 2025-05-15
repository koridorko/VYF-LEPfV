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


def calculate_normalised_grads(frames, step):
    """
    This method computes average gradients between frames in the video and normalizes them.
    """
    num_frames = len(frames)
    grads = []
    for i in range(0, num_frames, step):
        R, G, B = cv2.split(frames[i].astype('float32'))
        R_prev, G_prev, B_prev = cv2.split(frames[i-step].astype('float32'))
        R_grad = cv2.absdiff(R, R_prev)
        G_grad = cv2.absdiff(G, G_prev)
        B_grad = cv2.absdiff(B, B_prev)
        grads.append(cv2.merge((R_grad, G_grad, B_grad)))

    # Normalize the gradients to the range [0, 255]
    grads = np.array(grads)
    grads_normalized = cv2.normalize(grads, None, 0, 255, cv2.NORM_MINMAX)

    return grads_normalized

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

def colorful_gradient_threshold(frames, step, threshold=0.5):
    """
    This method normalised gradients between frames in the video and if the average gradient
    for pixel is above percentage of the threshold, it is set to average color value of the pixel
    in the image.
    """
    avg = create_photo_avg(frames, False)
    grads = calculate_normalised_grads(frames, step)

    # blending images with threshold
    blended = np.zeros_like(avg)
    for i in range(grads.shape[0]):
        mask = grads[i] > threshold
        
        blended[mask] = avg[mask]
        blended[~mask] = grads[i][~mask]


    return blended.astype('uint8')

    
def create_photo_blend(frames, short, step_cmd, alpha=0.5):
    """
    Blenduje priemerovaný obraz s gradientovým pomocou váhy alpha.
    alpha=0.0 -> čisto priemerovanie
    alpha=1.0 -> čisto gradient
    """
    avg = create_photo_avg(frames, short)
    grad = create_photo_grad(frames, short, step_cmd)

    # float conversion so we can do blending
    avg_float = avg.astype('float32')
    grad_float = grad.astype('float32')

    # Blend
    blended = cv2.addWeighted(avg_float, 1.0 - alpha, grad_float, alpha, 0)

    return blended.astype('uint8')

def create_gaussian_pyramid(image, levels=5):
    pyramid = [image]
    for i in range(levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def create_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        upsampled = cv2.pyrUp(gaussian_pyramid[i + 1])

        # resize upsampled image to match the size of the current level
        if upsampled.shape != gaussian_pyramid[i].shape:
            upsampled = cv2.resize(upsampled, (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))

        laplacian = cv2.subtract(gaussian_pyramid[i], upsampled)
        laplacian_pyramid.append(laplacian)
    
    # append the last level of the Gaussian pyramid
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid

def blend_pyramids(avg_pyramid, derivates_pyramid, mask_pyramid, alpha=0.5):
    blended_pyramid = []
    for avg, deriv, mask in zip(avg_pyramid, derivates_pyramid, mask_pyramid):
        blended = mask * deriv + (1 - mask) * avg
        blended_pyramid.append(blended)

    return blended_pyramid

def reconstruct_from_pyramid(pyramid):
    image = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        # Upsample the image
        image = cv2.pyrUp(image)
        
        # Resize the upsampled image to match the current level
        if image.shape != pyramid[i].shape:
            image = cv2.resize(image, (pyramid[i].shape[1], pyramid[i].shape[0]))
        
        # Add the current level of the pyramid
        image = cv2.add(image, pyramid[i])
        
    return image

def create_mask_movement(grad_photo):
    """Vytvori masku pomocou gradientu"""
    sobel = cv2.Sobel(grad_photo, cv2.CV_64F, 1, 1, ksize=5)
    sobel = cv2.convertScaleAbs(sobel)
    greyscale_mask = cv2.cvtColor(sobel, cv2.COLOR_BGR2GRAY)
    # make mask rgb as first immage for same dimensions
    greyscale_mask = cv2.merge((greyscale_mask, greyscale_mask, greyscale_mask))
    # save mask as mask.png
    cv2.imwrite('./photos/mask.png', greyscale_mask)
    return greyscale_mask

def create_photo_blend_with_pyramids(frames, short, step_cmd, alpha=0.5, pyramid_levels=5):
    """
    Blenduje priemerovaný obraz a gradient pomocou Gaussovej a Laplacovej pyramídy s váhou alpha.
    alpha=0.0 -> čisto priemerovanie
    alpha=1.0 -> čisto gradient
    pyramid_levels = počet úrovní pyramídy
    """
    # creating average image
    avg = create_photo_avg(frames, short)
    # and derivative image
    grad = create_photo_grad(frames, short, step_cmd)

    # creating mask
    mask = create_mask_movement(grad)
    # normalisation of the mask
    mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX)

    mask_pyramid = create_gaussian_pyramid(mask, levels=pyramid_levels)

    # creating of gaussian and laplacian pyramids
    avg_pyramid = create_gaussian_pyramid(avg, levels=pyramid_levels)
    grad_pyramid = create_gaussian_pyramid(grad, levels=pyramid_levels)
    avg_pyramid = create_laplacian_pyramid(avg_pyramid)
    grad_pyramid = create_laplacian_pyramid(grad_pyramid)
    # Blend pyramids
    blended_pyramid = blend_pyramids(avg_pyramid, grad_pyramid, mask_pyramid, alpha=alpha)

    # reconstruct the blended image from the blended pyramid
    blended_image = reconstruct_from_pyramid(blended_pyramid)

    return blended_image.astype('uint8')


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
    cv2.imwrite(output_path, result_image)