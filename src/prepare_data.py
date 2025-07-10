import argparse
from pathlib import Path
import cv2
from cv2.typing import MatLike
import mediapipe as mp
import numpy as np
import os
import json
from tqdm import tqdm
from PIL import Image

import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.selfie_segmentation as mp_selfie_segmentation


def process_frame(
    frame: MatLike,
    i_frame: int,
    selfie_segmentation: mp_selfie_segmentation.SelfieSegmentation,
    face_mesh: mp_face_mesh.FaceMesh,
    pose_detection: mp_pose.Pose,
    *,
    image_size: tuple[int, int], # width, height
    out_face_hf: float,
    out_face_xf: float = 0.5,
    out_face_yf: float,
    clahe_grid_size: int = 0,
    debug: bool = False,
):
        
    frame_h, frame_w, c = frame.shape
    if c != 3:
        raise NotImplementedError("")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    nobg_frame_rgba = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
    nobg_frame_rgba[:, :, 0:3] = frame_rgb # RGBチャンネルをコピー

    # frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
    # frame_rgb = cv2.cvtCOlor(frame_rgba, cv2.COLOR_RGBA2RGB)
    
    # デバッグ用に元のフレームをコピー
    debug_display_frame = frame.copy()

    # --- 1. 背景セグメンテーション ---
    results_segmentation = selfie_segmentation.process(frame_rgb)
    segmentation_mask = getattr(results_segmentation, 'segmentation_mask')
    if segmentation_mask is None:
        if debug:
            print(f"Skipping frame {i_frame}: Segmentation mask not found.")
        return None

    # マスクをグレースケール8ビットに変換し、透明度チャンネルとして使用
    condition = np.stack((segmentation_mask,) * 3, axis=-1) > 0.1 # 閾値調整
    binary_mask = (condition * 255).astype(np.uint8)[:, :, 0] # 3チャンネルから1チャンネルに

    # RGBA画像を準備: 背景を透明にする
    nobg_frame_rgba[:, :, 3] = binary_mask # アルファチャンネルにマスクを適用

    if clahe_grid_size:
        nobg_frame_rgba = apply_clahe(nobg_frame_rgba, grid_size=(clahe_grid_size, clahe_grid_size))

    # デバッグ表示: セグメンテーションマスク
    # if debug:
    #     disp_nobg_frame_rgba = (nobg_frame_rgba[:, :, :3] * (nobg_frame_rgba[:, :, 3:] / 255.0)).astype(np.uint8)
    #     cv2.imshow(f'RGBA (Background Removed)', cv2.cvtColor(disp_nobg_frame_rgba, cv2.COLOR_RGB2BGR))
    #     cv2.waitKey(0)
        # cv2.destroyWindow(f'RGBA (Background Removed)')


    # --- 2. 顔とポーズの検出 ---
    results_face_mesh = face_mesh.process(frame_rgb)
    results_pose = pose_detection.process(frame_rgb)

    if not results_face_mesh.multi_face_landmarks: # type: ignore
        if debug:
            print(f"Skipping frame {i_frame}: Face landmarks not found.")
        return None
    
    if len(results_face_mesh.multi_face_landmarks) > 1: # type: ignore
        if debug:
            print(f"Skipping frame {i_frame}: Multiple Face found.")
        return None

    if not results_pose.pose_landmarks: # type: ignore
        if debug:
            print(f"Skipping frame {i_frame}: Pose landmarks not found.")
        return None

    raw_face_landmarks = results_face_mesh.multi_face_landmarks[0].landmark # type: ignore
    face_landmarks = np.zeros((len(raw_face_landmarks), 3))
    for i, landmark in enumerate(raw_face_landmarks):
        face_landmarks[i] = [landmark.x, landmark.y, landmark.z]

    raw_pose_landmarks = results_pose.pose_landmarks.landmark # type: ignore
    pose_landmarks = np.zeros((len(raw_pose_landmarks), 3))
    pose_confidences = np.zeros((len(raw_pose_landmarks),))
    for i, landmark in enumerate(raw_pose_landmarks):
        pose_landmarks[i] = [landmark.x, landmark.y, landmark.z]
        pose_confidences[i] = getattr(landmark, 'visibility', np.nan)
    
    pad_ratio = 0.1
    out_w, out_h = image_size

    face_xf_min, face_yf_min, _ = np.min(face_landmarks, axis=0) # [0.0, 1.0]
    face_xf_max, face_yf_max, _ = np.max(face_landmarks, axis=0) # [0.0, 1.0]
    # pose_xf_min, pose_yf_min, _ = np.min(pose_landmarks, axis=0) # [0.0, 1.0]
    # pose_xf_max, pose_yf_max, _ = np.max(pose_landmarks, axis=0) # [0.0, 1.0]
    face_xf = (face_xf_min + face_xf_max) / 2
    face_yf = (face_yf_min + face_yf_max) / 2
    # face_wf = face_xf_max - face_xf_min # [0.0, 1.0]
    face_hf = face_yf_max - face_yf_min # [0.0, 1.0]

    scale = out_face_hf / face_hf
    trans_x = out_face_xf - face_xf * scale # face_xf * scale + trans_x = out_face_xf
    trans_y = out_face_yf - face_yf * scale # face_yf * scale + trans_y = out_face_yf

    # x_max, y_max = max(face_x_max, pose_x_max), max(face_y_max, pose_y_max)
    # ctr_x = int((x_min + x_max) / 2)
    # ctr_y = int((y_min + y_max) / 2)
    # box_w: float = (x_max - x_min) * (1 + 2 * pad_ratio)
    # box_h: float = (y_max - y_min) * (1 + 2 * pad_ratio)
    
    # box_out_r = max(box_w / out_w, box_h / out_h)
    # crop_w = int(round((out_w * box_out_r) / 2)) * 2
    # crop_h = int(round((out_h * box_out_r) / 2)) * 2
    # # crop_w = box_w
    # # crop_h = box_h

    # crop_bx = ctr_x - crop_w // 2
    # crop_ex = ctr_x + crop_w // 2
    # crop_by = ctr_y - crop_h // 2
    # crop_ey = ctr_y + crop_h // 2

    # src_crop_bx = max(0, crop_bx)
    # src_crop_ex = min(frame_w, crop_ex)
    # src_crop_by = max(0, crop_by)
    # src_crop_ey = min(frame_h, crop_ey)

    # if not (src_crop_bx < src_crop_ex and src_crop_by < src_crop_ey):
    #     if debug:
    #         print(f"Skipping frame {i_frame}: bbox is out of range.")
    #     return None

    # cropped_pil_from_src = Image.fromarray(nobg_frame_rgba.astype(np.uint8)[src_crop_by:src_crop_ey, src_crop_bx:src_crop_ex])
    # cropped_frame_rgba = Image.new('RGBA', (crop_w, crop_h), (0, 0, 0, 0))
    # cropped_frame_rgba.paste(cropped_pil_from_src, (src_crop_bx - crop_bx, src_crop_by - crop_by))
    # out_frame_rgba = cropped_frame_rgba.resize(image_size)
    out_frame = Image.new('RGBA', (out_w, out_h), (0, 0, 0, 0))
    pil_nobg = Image.fromarray(nobg_frame_rgba)
    pil_nobg = pil_nobg.resize((int(out_w * scale), int(out_h * scale)))
    out_frame.paste(pil_nobg, (int(pil_nobg.width * trans_x), int(pil_nobg.height * trans_y)))

    normalized_x = face_landmarks[:, 0] * scale + trans_x
    normalized_y = face_landmarks[:, 1] * scale + trans_y
    normalized_face_landmarks = np.c_[normalized_x, normalized_y, face_landmarks[:, 2]]

    normalized_x = pose_landmarks[:, 0] * scale + trans_x
    normalized_y = pose_landmarks[:, 1] * scale + trans_y
    normalized_pose_landmarks = np.c_[normalized_x, normalized_y, pose_landmarks[:, 2]]

    # デバッグ表示: クロップ＆リサイズ後の画像と正規化されたキーポイント
    if debug:
        debug_out_frame = cv2.cvtColor(np.array(out_frame), cv2.COLOR_RGBA2BGR)
        
        # 正規化されたキーポイントを描画 (OUTPUT_IMAGE_SIZE にスケール)
        for l in normalized_face_landmarks:
            x = int(l[0] * image_size[0])
            y = int(l[1] * image_size[1])
            cv2.circle(debug_out_frame, (x, y), 2, (0, 255, 0), -1) # 顔は緑
        for l in normalized_pose_landmarks:
            x = int(l[0] * image_size[0])
            y = int(l[1] * image_size[1])
            cv2.circle(debug_out_frame, (x, y), 3, (255, 0, 0), -1) # ポーズは青
            
        # 描画されたバウンディングボックスも表示 (元のフレーム上)
        debug_disp_frame = frame.copy()
        # cv2.rectangle(debug_disp_frame, (int(face_x_min), int(face_y_min)), (int(face_x_max), int(face_y_max)), (0, 255, 255), 2)
        # cv2.rectangle(debug_disp_frame, (int(pose_x_min), int(pose_y_min)), (int(pose_x_max), int(pose_y_max)), (0, 0, 255), 2)
        # cv2.rectangle(debug_disp_frame, (crop_bx, crop_by), (crop_ex, crop_ey), (0, 128, 255), 2)

        mp_drawing.draw_landmarks(
            image=debug_disp_frame,
            landmark_list=results_face_mesh.multi_face_landmarks[0], # type: ignore
            connections=mp_face_mesh.FACEMESH_TESSELATION, # type: ignore
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
        )
        mp_drawing.draw_landmarks(
            image=debug_disp_frame,
            landmark_list=results_pose.pose_landmarks, # type: ignore
            connections=mp_pose.POSE_CONNECTIONS, # type: ignore
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
        )

        cv2.imshow(f'Processed Image with Normalized Landmarks', debug_out_frame)
        cv2.imshow(f'Original with Bounding Box', debug_disp_frame)
        cv2.waitKey(0)
        # cv2.destroyWindow(f'Processed Image with Normalized Landmarks')
        # cv2.destroyWindow(f'Original with Bounding Box')


    # --- 5. 復元パラメータの記録 ---
    restoration_params = {
        'frame_w': frame_w,
        'frame_h': frame_h,
        'scale': scale,
        'trans_x': trans_x,
        'trans_y': trans_y,
        # 'crop_bx': crop_bx,
        # 'crop_by': crop_by,
        # 'crop_w': crop_w,
        # 'crop_h': crop_h,
        'out_w': out_w,
        'out_h': out_h,
    }

    # --- 最終的なアノテーションデータ ---
    annotation_data = {
        'frame_idx': i_frame,
        'normalized_face_landmarks': normalized_face_landmarks,
        'normalized_pose_landmarks': normalized_pose_landmarks,
        'pose_confidences': pose_confidences,
        'restoration_params': json.dumps(restoration_params),
    }

    return out_frame, annotation_data



def prepare_dataset_from_video(
    video_path: Path | str, # 入力動画ファイルのパス
    output_dir: Path | str, # 処理済みデータセットの出力ディレクトリ
    image_size: tuple[int, int], # 出力画像のサイズ (例: 128x128)
    out_face_hf: float,
    out_face_yf: float,
    skip_frames: int,            # 何フレームごとに処理するか (例: 1で全フレーム)
    frame_start: int = 0,
    frame_limit: int | None = None, # 上限フレーム（このフレームまで処理する） 
    debug: bool = False,    # デバッグモードを有効にするか (True/False)
    clahe_grid_size: int = 0,    # CLAHE Gird size (0: disable CLAHE)
    pose_model_type: int = 2,
):
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i_frame_end = (frame_limit+1) if frame_limit is not None else total_frames

    output_dir.mkdir(exist_ok=True, parents=True)
    images_output_dir = output_dir / 'images'
    annotations_output_dir = output_dir / 'annotations'
    images_output_dir.mkdir(exist_ok=True)
    annotations_output_dir.mkdir(exist_ok=True)

    # Mediapipeのモデルはコンテキストマネージャーで適切にクローズする
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation, \
        mp_face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True) as face_mesh, \
        mp_pose.Pose(model_complexity=pose_model_type) as pose_detection:

        processed_count = 0
        for i_frame in tqdm(range(i_frame_end)):
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {i_frame}")

            if i_frame < frame_start or i_frame % skip_frames != 0:
                continue

            result = process_frame(frame, i_frame, selfie_segmentation, face_mesh, pose_detection, out_face_hf=out_face_hf, out_face_yf=out_face_yf, image_size=image_size, clahe_grid_size=clahe_grid_size, debug=debug)
            if result is None:
                continue
            out_frame, annotation_data = result

            # Save
            out_frame.save(images_output_dir / f"frame_{i_frame:06d}.png")
            np.savez_compressed(annotations_output_dir / f"frame_{i_frame:06d}.npz", **annotation_data)
            
            processed_count += 1

    print(f"Processing complete. Processed {processed_count} frames.")
    cap.release()
    cv2.destroyAllWindows() # すべてのウィンドウを閉じる
    print(f"Dataset created in {output_dir}")


def apply_clahe(frame: MatLike, *, clip_limit=2.0, grid_size=(8, 8)):
    """
    Apply clahe to normalize brightness and contrast

    Args:
        frame_rgba (np.array): Input frame. RGBA (4ch) or BGR (3ch)
        clip_limit (float): コントラスト制限の閾値。小さいほどコントラスト制限が強くなる。
        tile_grid_size (tuple): グリッドのサイズ (例: (8, 8))。

    Returns:
        np.array: CLAHE-applied frame
    """
    if frame.shape[2] == 4: # RGBA
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    else: # BGR
        bgr_frame = frame

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    
    ycrcb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YCrCb)
    ycrcb_frame[:, :, 0] = clahe.apply(ycrcb_frame[:, :, 0])
    enhanced_bgr_frame = cv2.cvtColor(ycrcb_frame, cv2.COLOR_YCrCb2BGR)

    if frame.shape[2] == 4:
        enhanced_rgba_frame = cv2.cvtColor(enhanced_bgr_frame, cv2.COLOR_BGR2RGBA)
        enhanced_rgba_frame[:, :, 3] = frame[:, :, 3] # Copy Alpha channel
        return enhanced_rgba_frame
    else:
        return enhanced_bgr_frame


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset from video using MediaPipe.")
    parser.add_argument('video_path', type=str, help='Path to the input video file.')
    parser.add_argument('output_dir', type=str, help='Directory to save the processed images and data files.')
    parser.add_argument('-imgw', '--image_width', type=int, required=True, help='Width for the saved images.')
    parser.add_argument('-imgh', '--image_height', type=int, required=True, help='Height for the saved images.')
    parser.add_argument('-skip', '--skip-frames', type=int, default=1, help='Skip frames (1: process all frames).')
    parser.add_argument('-start', '--frame-start', type=int, default=0, help='Start frame index (0: first).')
    parser.add_argument('-limit', '--frame-limit', type=int, default=None, help='The last frame index to process (None: process all frames).')
    parser.add_argument('-pmtype', '--pose-model-type', type=int, choices=[1, 2], default=2, help='Pose model type (1:fast, 2: detail)')
    parser.add_argument('-fh', '--out-face-hf', type=float, required=True, help='Height (0-1 float scale) of face in output image')
    parser.add_argument('-fy', '--out-face-yf', type=float, required=True, help='Y Positon (0-1 float scale) of face in output image')
    parser.add_argument('-clhsize', '--clahe-grid-size', type=int, default=8, help='CLAHE grid size (0: disable CLAHE)')
    parser.add_argument('-debug', '--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()

    prepare_dataset_from_video(
        args.video_path,
        args.output_dir,
        image_size=(args.image_width, args.image_height),
        skip_frames=args.skip_frames,
        frame_start=args.frame_start,
        frame_limit=args.frame_limit,
        out_face_hf=args.out_face_hf,
        out_face_yf=args.out_face_yf,
        clahe_grid_size=args.clahe_grid_size,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
