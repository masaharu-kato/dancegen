import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
from tqdm import tqdm

def prepare_dataset_from_video(video_path, output_dir, image_size=(128, 128)):
    """
    動画ファイルからMediaPipeで顔メッシュとポーズ情報を抽出し、
    画像ファイルとNumpyデータファイルとして保存するスクリプト。

    Args:
        video_path (str): 入力動画ファイルのパス。
        output_dir (str): 出力データを保存するディレクトリ。
        image_size (tuple): 保存する画像のサイズ (width, height)。
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # MediaPipeの初期化
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # 顔メッシュとポーズモデルのインスタンス化
    # max_num_faces=1: 単一人物を想定
    # min_detection_confidence/min_tracking_confidence: 検出・追跡の閾値
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1, # 0, 1, 2. 2 is most accurate but slower.
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # 動画のキャプチャ
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video: {video_path} with {frame_count} frames.")

    processed_frames = 0
    for frame_idx in tqdm(range(frame_count), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe処理のためにBGEからRGBに変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 顔メッシュとポーズの検出
        face_results = face_mesh.process(frame_rgb)
        pose_results = pose.process(frame_rgb)

        face_coords = []
        if face_results.multi_face_landmarks:
            # 最初の顔メッシュのみを使用 (max_num_faces=1なので)
            for landmark in face_results.multi_face_landmarks[0].landmark:
                # x, y, z はそれぞれ画像の幅、高さに対する相対座標 (0.0-1.0)
                face_coords.extend([landmark.x, landmark.y, landmark.z])
        
        pose_coords = []
        if pose_results.pose_landmarks:
            for landmark in pose_results.pose_landmarks.landmark:
                # x, y, z はそれぞれ画像の幅、高さに対する相対座標 (0.0-1.0)
                # visibility はランドマークの視認性
                pose_coords.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                
        # 顔メッシュとポーズデータが両方存在する場合のみ保存
        # (片方だけでも保存する場合は条件を調整してください)
        if face_coords and pose_coords:
            # 結合してNumPy配列として保存
            # 注意: ポーズのvisibilityも含むため、次元数が変わります。
            # 468*3 (face) + 33*4 (pose) = 1404 + 132 = 1536 次元
            combined_data = np.array(face_coords + pose_coords, dtype=np.float32)
            
            # 画像の保存
            # トレーニング用に画像サイズをリサイズ
            resized_frame = cv2.resize(frame, image_size)
            
            # ファイル名を連番で生成
            # 例: frame_00000.png, frame_00000.npy
            frame_filename = f"frame_{processed_frames:05d}"
            image_save_path = os.path.join(output_dir, f"{frame_filename}.png")
            data_save_path = os.path.join(output_dir, f"{frame_filename}.npy")

            cv2.imwrite(image_save_path, resized_frame)
            np.save(data_save_path, combined_data)
            processed_frames += 1

    cap.release()
    face_mesh.close()
    pose.close()
    print(f"Finished processing. Saved {processed_frames} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset from video using MediaPipe.")
    parser.add_argument('video_path', type=str, help='Path to the input video file.')
    parser.add_argument('output_dir', type=str, help='Directory to save the processed images and data files.')
    parser.add_argument('--image_width', type=int, default=128,
                        help='Width for the saved images.')
    parser.add_argument('--image_height', type=int, default=128,
                        help='Height for the saved images.')
    
    args = parser.parse_args()

    # 既存のデータディレクトリが存在する場合、上書きに注意を促す
    if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)) > 0:
        print(f"Warning: Output directory '{args.output_dir}' is not empty.")
        print("Existing files might be overwritten or combined.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            exit()

    prepare_dataset_from_video(
        args.video_path,
        args.output_dir,
        image_size=(args.image_width, args.image_height)
    )