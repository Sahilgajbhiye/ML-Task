import cv2
from ultralytics import YOLO
import os

def evaluate_reid(video_path):
    model = YOLO("yolov8n.pt")  

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    player_id_map = {}  
    next_player_id = 0
    
    
    total_frames = 0
    re_identification_count = 0
    
    active_players = {}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        total_frames += 1

        results = model.track(frame, persist=True, classes=[0])

        current_frame_yolo_track_ids = set()
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            yolo_track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, yolo_track_id in zip(boxes, yolo_track_ids):
                current_frame_yolo_track_ids.add(yolo_track_id)

                if yolo_track_id not in player_id_map:
                    
                    re_identified = False
                    for p_id, p_info in list(active_players.items()): 
                        
                        if p_info['status'] == 'lost':
                            player_id_map[yolo_track_id] = p_id
                            active_players[p_id]['status'] = 'active'
                            re_identification_count += 1
                            re_identified = True
                            break

                    if not re_identified:
                        player_id_map[yolo_track_id] = next_player_id
                        active_players[next_player_id] = {'status': 'active', 'last_box': box}
                        next_player_id += 1
                
                active_players[player_id_map[yolo_track_id]]['last_box'] = box
                active_players[player_id_map[yolo_track_id]]['status'] = 'active'

        for p_id in list(active_players.keys()):
            found_in_current_frame = False
            for yolo_track_id, mapped_id in player_id_map.items():
                if mapped_id == p_id and yolo_track_id in current_frame_yolo_track_ids:
                    found_in_current_frame = True
                    break
            if not found_in_current_frame:
                active_players[p_id]['status'] = 'lost'

    cap.release()
    cv2.destroyAllWindows()

    print(f"Total frames processed: {total_frames}")
    print(f"Total unique players identified: {next_player_id}")
    print(f"Estimated re-identification events (simple check): {re_identification_count}")

if __name__ == "__main__":
    video_file = os.path.join(os.getcwd(), "15sec_input_720p.mp4")
    evaluate_reid(video_file)


