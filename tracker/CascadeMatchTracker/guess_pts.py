import yaml
import os
import math


class PointGuesser:
    def __init__(self, config_path = None):
        self.trajectory = []
        self.flag = False

        # 加载配置文件
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, '..', '..', 'config', 'guess_pts.yaml')
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # 初始化成员变量
        self.cos_factor = config.get('cos_factor', 0.003)  # 默认值为0.003
        self.d_factor = config.get('d_factor', 0.1)       # 默认值为0.01
        self.name_id_convert = {
            0: 'R1',
            1: 'R2',
            2: 'R3',
            3: 'R4',
            4: 'R7',
            5: 'B1',
            6: 'B2',
            7: 'B3',
            8: 'B4',
            9: 'B7',
        }

    def load_guess_points(self, config_path=None):
        """Load guess points from YAML configuration file."""
        if config_path is None:
            # Default path relative to the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, '..', '..', 'config', 'guess_pts.yaml')
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        return config['guess_points']

    def get_guess_points_for_robot(self, robot_name, ref_color):
        """Get guess points for a specific robot (e.g., 'R1').
        If ref_color is 'blue', mirror all guess points around the center (14, 7.5) of a 28x15 field.
        """
        guess_points = self.load_guess_points().get(robot_name, [])
        if ref_color == 'blue':
            center_x, center_y = 14, 7.5
            mirrored_points = []
            for x, y in guess_points:
                mirrored_x = 2 * center_x - x
                mirrored_y = 2 * center_y - y
                mirrored_points.append((mirrored_x, mirrored_y))
            return mirrored_points
        return guess_points
    
    def get_kalman_states(self, track):
        kalman_states = {}
        # if track.is_active and track.state.value in ['CONFIRMED', 'LOST']:
        pos_2d, vel_2d = track.kalman_filter_2d.get_state()
        kalman_states = {
            'pos_2d': pos_2d,
            'vel_2d': vel_2d,
        }
        return kalman_states
    
    def predict_points(self, track, ref_color):
        guess_points = self.get_guess_points_for_robot(self.name_id_convert[track.class_id], ref_color)
        kalman_states = self.get_kalman_states(track)

        #计算位置
        last_pos = kalman_states['pos_2d']
        #计算速度向量
        v_vec = (kalman_states['vel_2d'][0], kalman_states['vel_2d'][1])

        scores = []

        for point in guess_points:
            # 计算到固定点的向量
            d_vector = (point[0] - last_pos[0], point[1] - last_pos[1])

            # 计算余弦相似度
            dot_product = v_vec[0] * d_vector[0] + v_vec[1] * d_vector[1]
            v_norm = math.sqrt(v_vec[0] ** 2 + v_vec[1] ** 2)
            d_norm = math.sqrt(d_vector[0] ** 2 + d_vector[1] ** 2)
            cos_sim = dot_product / (v_norm * d_norm + 1e-8)  # 避免除零

            # 计算欧式距离
            distance = d_norm
            d_score = math.exp(-distance * self.d_factor)

            # 分数值确定优先级
            score = self.cos_factor * cos_sim + (1 - self.cos_factor) * d_score
            scores.append((point, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[0][0]
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    predictor = PointGuesser()
    robot_name = 'R1'
    guess_points = predictor.get_guess_points_for_robot(robot_name, 'blu')
    print(f"Guess points for {robot_name}: {guess_points}")

    # 假设消失点和速度
    last_pos = (9.00, 10.00)
    v_vec = (3, -10)
    # 计算分数
    scores = []
    for point in guess_points:
        d_vector = (point[0] - last_pos[0], point[1] - last_pos[1])
        dot_product = v_vec[0] * d_vector[0] + v_vec[1] * d_vector[1]
        v_norm = math.sqrt(v_vec[0] ** 2 + v_vec[1] ** 2)
        d_norm = math.sqrt(d_vector[0] ** 2 + d_vector[1] ** 2)
        cos_sim = dot_product / (v_norm * d_norm + 1e-8)
        distance = d_norm
        d_score = math.exp(-distance * predictor.d_factor)
        print (cos_sim, d_score)
        score = predictor.cos_factor * cos_sim + (1 - predictor.cos_factor) * d_score
        scores.append((point, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    best_point = scores[0][0]
    print(f"Predicted point: {best_point}")

    # 可视化
    plt.figure(figsize=(8, 6))
    plt.scatter([p[0] for p in guess_points], [p[1] for p in guess_points], c='blue', label='Guess Points')
    plt.scatter(last_pos[0], last_pos[1], c='red', label='Last Position')
    plt.arrow(last_pos[0], last_pos[1], v_vec[0]*0.1, v_vec[1]*0.1, head_width=0.1, color='red', label='Velocity')
    plt.scatter(best_point[0], best_point[1], c='green', label='Predicted Point', s=100, marker='*')
    plt.legend()
    plt.title(f'Guess Points Visualization for {robot_name}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()




    

    
        