import matplotlib.pyplot as plt
import numpy as np

def visualize_instance(instance_data, title="Data Instance Visualization"):
    """
    Visualizes a single data instance including the ego vehicle, surrounding vehicles,
    lane markings, and the goal target.

    Expected structure for `instance_data` dictionary:
    {
        'ego_vehicle': {'pos': [x, y], 'heading': angle_rad, 'length': l, 'width': w},
        'other_vehicles': [{'pos': [x, y], 'heading': angle_rad}, ...],
        'lanes': [[ [x1, y1], [x2, y2], ... ], ...],  # List of lane centerline/boundary coordinates
        'goal_target': [x, y]
    }
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. Plot Lanes / Road Network
    if 'lanes' in instance_data and instance_data['lanes']:
        for i, lane in enumerate(instance_data['lanes']):
            lane_pts = np.array(lane)
            # Label only once for cleaner legend
            label = 'Lanes' if i == 0 else None
            ax.plot(lane_pts[:, 0], lane_pts[:, 1], color='gray', linestyle='--', linewidth=1.5, label=label)

    # 2. Plot Other Vehicles
    if 'other_vehicles' in instance_data and instance_data['other_vehicles']:
        for i, veh in enumerate(instance_data['other_vehicles']):
            vx, vy = veh['pos']
            label = 'Other Vehicles' if i == 0 else None
            ax.scatter(vx, vy, color='royalblue', s=120, zorder=3, label=label)
            
            # Optional: Add heading direction arrow for other vehicles
            if 'heading' in veh:
                dx = np.cos(veh['heading']) * 2.0
                dy = np.sin(veh['heading']) * 2.0
                ax.arrow(vx, vy, dx, dy, head_width=0.8, head_length=1.0, fc='royalblue', ec='royalblue', zorder=3)

    # 3. Plot Goal Target
    if 'goal_target' in instance_data and instance_data['goal_target'] is not None:
        gx, gy = instance_data['goal_target']
        ax.scatter(gx, gy, color='limegreen', marker='*', s=250, edgecolor='black', zorder=4, label='Goal Target')

    # 4. Plot Main (Ego) Vehicle
    if 'ego_vehicle' in instance_data:
        ego = instance_data['ego_vehicle']
        ex, ey = ego['pos']
        ax.scatter(ex, ey, color='crimson', s=180, zorder=5, label='Main Car (Ego)')
        
        # Heading arrow for main car
        if 'heading' in ego:
            dx = np.cos(ego['heading']) * 3.0
            dy = np.sin(ego['heading']) * 3.0
            ax.arrow(ex, ey, dx, dy, head_width=1.0, head_length=1.2, fc='crimson', ec='crimson', zorder=5)

    # Formatting and Styling
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    plt.show()

# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":
    # Sample instance data
    sample_instance = {
        'ego_vehicle': {
            'pos': [0.0, 0.0],
            'heading': np.pi / 4  # 45 degrees in radians
        },
        'other_vehicles': [
            {'pos': [10.0, 12.0], 'heading': np.pi / 4},
            {'pos': [-5.0, 8.0], 'heading': np.pi / 2},
            {'pos': [15.0, -2.0], 'heading': 0.0}
        ],
        'lanes': [
            [[-10, -5], [0, 5], [10, 15], [20, 25]],  # Left lane
            [[-5, -10], [5, 0], [15, 10], [25, 20]]   # Right lane
        ],
        'goal_target': [18.0, 22.0]
    }

    # Call the visualization function
    visualize_instance(sample_instance, title="Autonomous Driving Data Instance #001")