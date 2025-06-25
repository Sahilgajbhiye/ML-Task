import cv2
from ultralytics import YOLO

def reid_players(video_path):
    model = YOLO("yolov8n.pt")  

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    player_tracks = {}
    next_player_id = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, classes=[0]) 

        
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        annotated_frame = results[0].plot()

        
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            if track_id not in player_tracks:
                player_tracks[track_id] = next_player_id
                next_player_id += 1
            
            
            cv2.putText(annotated_frame, f"Player ID: {player_tracks[track_id]}", (int(x - w / 2), int(y - h / 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Player Re-Identification", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_file = "15sec_input_720p.mp4"
    reid_players(video_file)


