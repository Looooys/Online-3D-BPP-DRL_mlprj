import json
import torch
import numpy as np

def convert_position(grid_pos, container_length, container_width):
    """
    Convert 10x10 grid position to actual container position
    
    Args:
        grid_pos: Position in 10x10 grid (0-99)
        container_length: Actual container length 
        container_width: Actual container width
        
    Returns:
        (x, y): Actual position coordinates in container
    """
    # Get x,y coordinates in 10x10 grid
    grid_x = grid_pos % 10
    grid_y = grid_pos // 10
    
    # Convert to actual container coordinates
    actual_x = (grid_x * container_length) / 10
    actual_y = (grid_y * container_width) / 10
    
    return (actual_x, actual_y)

def recalculate_ratio(boxes, container, used_containers):
    """
    Recalculate space utilization ratio
    
    Args:
        boxes: List of box dimensions (l,w,h) as lists
        container: Container dimensions (l,w,h) as list
        used_containers: Number of containers used
        
    Returns:
        float: New utilization ratio (0.0 to 1.0)
    """
    if used_containers == 0:
        return 0.0
    
    # Print debug information
    print(f"Boxes: {boxes}")
    print(f"Container: {container}")
    print(f"Used containers: {used_containers}")
        
    # Calculate total volume of all boxes
    box_volumes = [box[0] * box[1] * box[2] for box in boxes]
    total_box_volume = sum(box_volumes)
    
    # Calculate total volume of all containers
    container_volume = container[0] * container[1] * container[2]
    total_container_volume = container_volume * used_containers
    
    # Print volumes for verification
    print(f"Total box volume: {total_box_volume}")
    print(f"Total container volume: {total_container_volume}")
    print(f"Individual box volumes: {box_volumes}")
    print(f"Single container volume: {container_volume}")
    
    ratio = total_box_volume / total_container_volume
    return min(ratio, 1.0)  # Ensure ratio doesn't exceed 1.0

def process_data():
    # Load data
    pt_data = torch.load('converted_task3.pt')
    with open('results_scaled_task3_added_5preview.json', 'r') as f:
        json_data = json.load(f)
    
    # Process each group
    new_json_data = []
    for group_data in json_data:
        group = group_data['group']
        sequence = group_data['sequence']
        box_count = group_data['box_count']
        
        # Get corresponding PT data (8 items per group)
        start_idx = (group - 1) * 8
        pt_group_data = pt_data[start_idx:start_idx + 8]
        
        if len(pt_group_data) > 0:
            # Get the specific data item based on sequence
            selected_data = pt_group_data[sequence - 1]
            
            # Last element is container dimensions
            container_dims = selected_data[-1]
            
            # Get box dimensions (all elements except the last one)
            boxes = selected_data[:-1]
        else:
            continue
            
        # Count unique containers used
        container_nums = set()
        for action in group_data['container_actions']:
            container_nums.add(action[0])
        used_containers = len(container_nums)
        
        # Recalculate ratio
        new_ratio = recalculate_ratio(boxes, container_dims, used_containers)
        
        # Convert positions and add to actions
        new_actions = []
        for action in group_data['container_actions']:
            container_num = action[0]
            grid_pos = action[1]
            actual_pos = convert_position(grid_pos, container_dims[0], container_dims[1])
            
            new_actions.append({
                'container': container_num,
                'grid_position': grid_pos,
                'actual_position': {
                    'x': float(actual_pos[0]),
                    'y': float(actual_pos[1])
                }
            })
            
        # Create new group data
        new_group_data = {
            'group': group,
            'sequence': sequence,
            'original_ratio': group_data['ratio'],
            'recalculated_ratio': float(new_ratio),
            'box_count': box_count,
            'used_containers': used_containers,
            'container_dimensions': {
                'length': float(container_dims[0]),
                'width': float(container_dims[1]),
                'height': float(container_dims[2])
            },
            'actions': new_actions
        }
        
        new_json_data.append(new_group_data)
    
    # Save new JSON
    with open('processed_results.json', 'w') as f:
        json.dump(new_json_data, f, indent=2)

if __name__ == '__main__':
    process_data()