from time import clock
from acktr.model_loader import nnModel
from acktr.reorder import ReorderTree
import gym
import copy
from gym.envs.registration import register
from acktr.arguments import get_args
import json

def run_sequence(nmodel, raw_env, preview_num, c_bound):
    env = copy.deepcopy(raw_env)
    container_size = [10, 10, 10]  # 固定容器尺寸
    used_containers = 1
    total_box_volume = 0  # 已放入箱子的总体积
    container_volume = container_size[0] * container_size[1] * container_size[2]  # 单个容器体积
    
    # 获取整个序列的箱子
    all_boxes = env.box_creator.preview(500)  # 假设最大序列长度为500
    box_index = 0
    
    # 检查箱子尺寸是否超过容器
    for i, box in enumerate(all_boxes):
        if (box[0] > container_size[0] or 
            box[1] > container_size[1] or 
            box[2] > container_size[2]):
            print(f"Box size {box} exceeds container size {container_size}")
            return 0, 0, 0, 0, []
    
    if not all_boxes:
        return 0, 0, 0, 0, []
    
    default_counter = 0
    box_counter = 0
    start = clock()
    current_container_volume = 0
    actions = []
    container_actions = []  # 新增：用于跟踪容器-动作对
    
    while box_index < len(all_boxes):
        remaining_boxes = len(all_boxes) - box_index
        current_preview = min(preview_num, remaining_boxes)
        box_list = all_boxes[box_index:box_index + current_preview]
        print(f"Processing boxes: {box_list}")
        
        if not box_list:
            break
            
        # 检查是否遇到结束标记序列
        if len(box_list) >= 3:
            if (box_list[0] == [10, 10, 10] and 
                box_list[1] == [10, 10, 10] and 
                box_list[2] == (10, 10, 10)):
                print("Found end marker sequence")
                break
            
        tree = ReorderTree(nmodel, box_list, env, times=100)
        act, val, default = tree.reorder_search()
        obs, reward, done, info = env.step([act])
        
        box_counter += 1
        default_counter += int(default)
        
        # 获取当前尝试放入的箱子
        current_box = all_boxes[box_index]
        box_volume = current_box[0] * current_box[1] * current_box[2]
        
        if not done:  # 成功放置
            actions.append(act)
            container_actions.append((used_containers, act))  # 使用used_containers作为容器编号
            total_box_volume += box_volume
            current_container_volume += box_volume
        else:  # 当前容器放不下了
            print(f'Container {used_containers} utilization: {current_container_volume/container_volume:.4f}')
            if current_container_volume > 0:
                used_containers += 1
            current_container_volume = 0
            env.reset_space()
            
            # 在新容器中重新尝试放置
            tree = ReorderTree(nmodel, [current_box], env, times=100)
            act, val, default = tree.reorder_search()
            obs, reward, done, info = env.step([act])
            
            if not done:  # 在新容器中成功放置
                actions.append(act)
                container_actions.append((used_containers, act))  # 使用used_containers作为容器编号
                total_box_volume += box_volume
                current_container_volume += box_volume
            else:  # 在新容器中也无法放置
                print(f"Failed to place box {current_box} in new container")
                actions.append(0)
                container_actions.append((used_containers, 0))  # 使用used_containers作为容器编号
                total_box_volume += box_volume
                current_container_volume += box_volume
                print(f'Container {used_containers} utilization: {current_container_volume/container_volume:.4f}')
                used_containers += 1
                current_container_volume = 0
                env.reset_space()
                
        box_index += 1  # 移动到下一个箱子
        
    end = clock()
    time_cost = end - start
    
    used_containers = max(0, used_containers - 1)
    
    if used_containers <= 0:
        return 0, 0, 0, 0, []
    
    total_container_volume = used_containers * container_volume
    final_ratio = total_box_volume / total_container_volume if total_container_volume > 0 else 0
    print(f'Total time cost: {time_cost}')
    print(f'Total containers used: {used_containers}')
    print(f'Total box volume: {total_box_volume}')
    print(f'Total container volume: {total_container_volume}')
    print(f'Overall utilization ratio: {final_ratio:.4f}')
    
    default_rate = default_counter/box_counter if box_counter > 0 else 0
    return final_ratio, box_counter, time_cost, default_rate, container_actions

def unified_test(url, args, pruning_threshold = 0.5):
    nmodel = nnModel(url, args)
    data_url = './dataset/' + args.data_name
    env = gym.make(args.env_name,
                    box_set=args.box_size_set,
                    container_size=args.container_size,
                    test=True, data_name=data_url,
                    enable_rotation=args.enable_rotation,
                    data_type=args.data_type)
    print('Env name: ', args.env_name)
    print('Data url: ', data_url)
    print('Model url: ', url)
    print('Case number: ', args.cases)
    print('pruning threshold: ', pruning_threshold)
    print('Known item number: ', args.preview)
    
    times = args.cases
    group_size = 8  # 每组8个序列
    num_groups = times // group_size
    
    total_ratio = 0.0
    total_counter = 0.0
    total_time = 0.0
    total_drate = 0.0
    
    c_bound = pruning_threshold
    
    # 存储每组的最佳结果
    group_best_results = []
    
    for g in range(num_groups):
        print(f'\nProcessing group {g+1}/{num_groups}')
        group_results = []
        
        # 处理每组的8个序列
        for i in range(group_size):
            if (g * group_size + i) % 10 == 0:
                print('case', g * group_size + i + 1)
            env.reset()
            env.box_creator.preview(500)
            ratio, counter, time, depen_rate, container_actions = run_sequence(nmodel, env, args.preview, c_bound)
            group_results.append((ratio, counter, time, depen_rate, container_actions))
        
        # 选择当前组最佳序列的结果
        best_idx = max(range(group_size), key=lambda i: group_results[i][0])
        best_result = group_results[best_idx]
        
        # 记录该组的最佳信息，包含容器-动作对
        group_best_info = {
            'group': int(g + 1),
            'sequence': int(best_idx + 1),
            'ratio': float(best_result[0]),
            'box_count': int(best_result[1]),
            'container_actions': [(int(c), int(a)) for c, a in best_result[4]]  # 存储容器-动作对
        }
        group_best_results.append(group_best_info)
        
        print(f'Group {g+1} best sequence info:')
        print(f'  Sequence number: {best_idx + 1}')
        print(f'  Space utilization: {best_result[0]:.4f}')
        print(f'  Number of boxes: {best_result[1]}')
        print(f'  Container-action pairs: {best_result[4]}')  # 修改输出格式
        
        total_ratio += best_result[0]
        total_counter += best_result[1]
        total_time += best_result[2]
        total_drate += best_result[3]

    print()
    print('All cases have been done!')
    print('----------------------------------------------')
    print('Group-wise best results summary:')
    for info in group_best_results:
        print(f'\nGroup {info["group"]}:')
        print(f'  Best sequence: {info["sequence"]}')
        print(f'  Space utilization: {info["ratio"]:.4f}')
        print(f'  Number of boxes: {info["box_count"]}')
        print(f'  Container-action pairs: {info["container_actions"]}')
    
    # 保存结果到JSON文件
    output_filename = f'results_{args.data_name.replace(".pt", "")}_{args.preview}preview.json'
    with open(output_filename, 'w') as f:
        json.dump(group_best_results, f, indent=2)
    print(f'\nResults have been saved to {output_filename}')
    
    print('----------------------------------------------')
    print('average space utilization: %.4f'%(total_ratio/num_groups))
    print('average put item number: %.4f'%(total_counter/num_groups))
    print('average sequence time: %.4f'%(total_time/num_groups))
    print('average time per item: %.4f'%(total_time/total_counter))
    print('----------------------------------------------')

def registration_envs():
    register(
        id='Bpp-v0',                                  # Format should be xxx-v0, xxx-v1
        entry_point='envs.bpp0:PackingGame',   # Expalined in envs/__init__.py
    )

if __name__ == '__main__':
    registration_envs()
    args = get_args()
    pruning_threshold = 0.5  # pruning_threshold (default: 0.5)
    unified_test('pretrained_models/default_cut_2.pt', args, pruning_threshold)
    # args.enable_rotation = True
    # unified_test('pretrained_models/rotation_cut_2.pt', args, pruning_threshold)

    