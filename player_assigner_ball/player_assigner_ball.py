import sys 
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance

class PlayerAssignerBall:
    def __init__(self):
        self.max_distance = 70

    def assign_ball_to_player(self, players, ball):
        ball_position = get_center_of_bbox(ball)

        min_distance = float('inf')
        closest_player = -1

        for player_id, player in players.items():
            #player_position = get_center_of_bbox(player['bbox'])
            player_bbox = player['bbox']
            #print(player_position)
            # distance = measure_distance(player_position, ball_position)
            left_distance = measure_distance([player_bbox[0], player_bbox[-1]], ball_position)
            right_distance = measure_distance([player_bbox[2], player_bbox[-1]], ball_position)

            distance = min(left_distance, right_distance)
            if distance < self.max_distance:
                if distance < min_distance:
                    min_distance = distance
                    closest_player = player_id
        
        return closest_player