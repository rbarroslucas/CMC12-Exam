import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd

sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def plot_results(t_hist, X_hist, U_hist, system, x_ref):
    df_states = pd.DataFrame(X_hist, columns=['x', 'x_dot', 'theta1', 'theta2', 'theta1_dot', 'theta2_dot'])
    df_states['time'] = t_hist
    
    with sns.axes_style("whitegrid"):
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('MPC Control - Double Inverted Pendulum', fontsize=20, fontweight='bold', y=0.98)
        
        state_info = [
            ('x', 'Cart Position', 'm', 'tab:blue'),
            ('x_dot', 'Cart Velocity', 'm/s', 'tab:blue'),
            ('theta1', 'Angle θ1', 'rad', 'tab:blue'),
            ('theta2', 'Angle θ2', 'rad', 'tab:blue'),
            ('theta1_dot', 'Angular Velocity θ1', 'rad/s', 'tab:blue'),
            ('theta2_dot', 'Angular Velocity θ2', 'rad/s', 'tab:blue')
        ]
        
        for i, (col, name, unit, color) in enumerate(state_info):
            row = i // 2
            col_idx = i % 2
            ax = axes[row, col_idx]
            
            sns.lineplot(data=df_states, x='time', y=col, ax=ax, color=color, linewidth=2.5, label='Current State')
            
            ax.axhline(y=x_ref[i], color='red', linestyle='--', linewidth=2, alpha=0.8, label='Reference')
            
            ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{name} ({unit})', fontsize=12, fontweight='bold')
            ax.set_title(f'{name}', fontsize=14, fontweight='bold', pad=10)
            ax.legend(frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3)
            
            ax.fill_between(df_states['time'], df_states[col], x_ref[i], alpha=0.2, color=color)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig("pendulum_states_mpc_seaborn.png", dpi=300, bbox_inches='tight')
    
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(14, 6))
        
        df_control = pd.DataFrame({
            'time': t_hist[:-1],
            'force': U_hist[:, 0]
        })
        
        sns.lineplot(data=df_control, x='time', y='force', ax=ax, 
                    color='steelblue', linewidth=2.5)
        
        ax.fill_between(df_control['time'], df_control['force'], 
                       alpha=0.3, color='steelblue')
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Control Force (N)', fontsize=14, fontweight='bold')
        ax.set_title('MPC Control Signal', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        stats_text = f'Max: {np.max(U_hist):.2f} N\nMin: {np.min(U_hist):.2f} N\nMean: {np.mean(U_hist):.2f} N'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
                facecolor='lightblue', alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        plt.savefig("control_signal_mpc_seaborn.png", dpi=300, bbox_inches='tight')

def plot_performance_metrics(t_hist, X_hist, x_ref):
    errors = np.linalg.norm(X_hist - x_ref, axis=1)
    position_error = np.abs(X_hist[:, 0] - x_ref[0])
    angle1_error = np.abs(X_hist[:, 2] - x_ref[2])
    angle2_error = np.abs(X_hist[:, 3] - x_ref[3])
    
    df_errors = pd.DataFrame({
        'time': t_hist,
        'total_error': errors,
        'position_error': position_error,
        'angle1_error': angle1_error,
        'angle2_error': angle2_error
    })
    
    with sns.axes_style("whitegrid"):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Metrics', fontsize=18, fontweight='bold', y=0.98)
        
        # Total error
        sns.lineplot(data=df_errors, x='time', y='total_error', ax=axes[0,0], 
                    color='red', linewidth=2.5)
        axes[0,0].fill_between(df_errors['time'], df_errors['total_error'], 
                              alpha=0.3, color='red')
        axes[0,0].set_title('Total Error (Norm)', fontweight='bold')
        axes[0,0].set_ylabel('Error', fontweight='bold')
        
        # Position error
        sns.lineplot(data=df_errors, x='time', y='position_error', ax=axes[0,1], 
                    color='blue', linewidth=2.5)
        axes[0,1].fill_between(df_errors['time'], df_errors['position_error'], 
                              alpha=0.3, color='blue')
        axes[0,1].set_title('Position Error', fontweight='bold')
        axes[0,1].set_ylabel('Error (m)', fontweight='bold')
        
        # Angle 1 error
        sns.lineplot(data=df_errors, x='time', y='angle1_error', ax=axes[1,0], 
                    color='green', linewidth=2.5)
        axes[1,0].fill_between(df_errors['time'], df_errors['angle1_error'], 
                              alpha=0.3, color='green')
        axes[1,0].set_title('Angle θ1 Error', fontweight='bold')
        axes[1,0].set_ylabel('Error (rad)', fontweight='bold')
        axes[1,0].set_xlabel('Time (s)', fontweight='bold')
        
        # Angle 2 error
        sns.lineplot(data=df_errors, x='time', y='angle2_error', ax=axes[1,1], 
                    color='orange', linewidth=2.5)
        axes[1,1].fill_between(df_errors['time'], df_errors['angle2_error'], 
                              alpha=0.3, color='orange')
        axes[1,1].set_title('Angle θ2 Error', fontweight='bold')
        axes[1,1].set_ylabel('Error (rad)', fontweight='bold')
        axes[1,1].set_xlabel('Time (s)', fontweight='bold')
        
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig("performance_metrics.png", dpi=300, bbox_inches='tight')


def animate_pendulum(system, X_hist, filename="pendulum_mpc_simulation.gif"):
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('MPC Simulation - Inverted Double Pendulum', fontsize=16, fontweight='bold')
    
    L1, L2 = system.L1, system.L2
    skip_frames = 3
    
    def animate(frame):
        ax.clear()
        
        if frame >= len(X_hist):
            return
            
        x, x_dot, theta1, theta2, theta1_dot, theta2_dot = X_hist[frame]
        
        x_cart = x
        y_cart = 0
        
        x1 = x_cart + L1 * np.sin(theta1)
        y1 = L1 * np.cos(theta1)
        
        x2 = x1 + L2 * np.sin(theta2)
        y2 = y1 + L2 * np.cos(theta2)
        
        ax.axhline(y=-0.15, color='saddlebrown', linewidth=8, alpha=0.7)
        
        track_length = 3
        ax.plot([-track_length, track_length], [-0.1, -0.1], 'k-', linewidth=4)
        
        cart_width = 0.25
        cart_height = 0.15
        cart = plt.Rectangle((x_cart - cart_width/2, y_cart - cart_height/2), 
                           cart_width, cart_height, fill=True, 
                           color='steelblue', alpha=0.8, linewidth=2, edgecolor='navy')
        ax.add_patch(cart)
        
        ax.plot([x_cart, x1], [y_cart, y1], 'o-', color='red', 
                linewidth=4, markersize=10, markerfacecolor='darkred', 
                markeredgecolor='black', markeredgewidth=2)
        ax.plot([x1, x2], [y1, y2], 'o-', color='green', 
                linewidth=4, markersize=10, markerfacecolor='darkgreen',
                markeredgecolor='black', markeredgewidth=2)
        
        if frame > 10:
            trace_start = max(0, frame - 20)
            x2_trace = [x + L1 * np.sin(X_hist[i, 2]) + L2 * np.sin(X_hist[i, 3]) 
                       for i in range(trace_start, frame)]
            y2_trace = [L1 * np.cos(X_hist[i, 2]) + L2 * np.cos(X_hist[i, 3]) 
                       for i in range(trace_start, frame)]
            ax.plot(x2_trace, y2_trace, 'b-', alpha=0.5, linewidth=2)
        
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-0.3, 2.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Animation - Frame {frame}', fontweight='bold')
        
        info_text = (f'Position: {x:.3f} m\n'
                    f'Velocity: {x_dot:.3f} m/s\n'
                    f'θ1: {theta1:.3f} rad ({np.degrees(theta1):.1f}°)\n'
                    f'θ2: {theta2:.3f} rad ({np.degrees(theta2):.1f}°)\n'
                    f'ω1: {theta1_dot:.3f} rad/s\n'
                    f'ω2: {theta2_dot:.3f} rad/s')
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
    
    frames = range(0, len(X_hist), skip_frames)
    anim = FuncAnimation(fig, animate, frames=frames, interval=50, blit=False)
    writer = PillowWriter(fps=20)
    anim.save("pendulum_mpc_simulation.gif", writer=writer)