import matplotlib.pyplot as plt
import numpy as np

def generate_comparison_figure():
    fig = plt.figure(figsize=(14, 9), dpi=300)
    
    # Color palette
    c_vgg = "#3498db"      # Blue
    c_eff = "#2ecc71"      # Green
    
    # 1. Global Metrics Bar Chart
    ax1 = plt.subplot(2, 2, 1)
    metrics = ['Accuracy', 'ROC-AUC', 'Macro F1', "Cohen's Kappa"]
    vgg_scores = [56.67, 74.17, 57.04, 35.00]
    eff_scores = [87.33, 94.46, 85.45, 78.00]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, vgg_scores, width, label='VGG16 (Baseline)', color=c_vgg, edgecolor='black', linewidth=0.8)
    rects2 = ax1.bar(x + width/2, eff_scores, width, label='EfficientNet-B0 (Optimized)', color=c_eff, edgecolor='black', linewidth=0.8)
    
    ax1.set_ylabel('Score (%) / Index', fontsize=11, fontweight='bold')
    ax1.set_title('Overall Performance Comparison', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=10, fontweight='bold')
    ax1.legend(frameon=True, facecolor='#f8f9fa', edgecolor='#bdc3c7')
    ax1.set_ylim(0, 110)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%' if h > 40 else f'{h/100:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%' if h > 40 else f'{h/100:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1e8449')
                    
    # 2. Per-Class F1-Score Comparison
    ax2 = plt.subplot(2, 2, 2)
    classes = ['Grade-1', 'Grade-2', 'Grade-3']
    vgg_f1 = [66.67, 44.44, 60.00]
    eff_f1 = [89.36, 83.17, 83.81]
    
    x2 = np.arange(len(classes))
    rects2_1 = ax2.bar(x2 - width/2, vgg_f1, width, label='VGG16', color=c_vgg, edgecolor='black', linewidth=0.8)
    rects2_2 = ax2.bar(x2 + width/2, eff_f1, width, label='EfficientNet-B0', color=c_eff, edgecolor='black', linewidth=0.8)
    
    ax2.set_ylabel('F1-Score (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Per-Class F1-Score Comparison', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(classes, fontsize=10, fontweight='bold')
    ax2.legend(frameon=True, facecolor='#f8f9fa', edgecolor='#bdc3c7')
    ax2.set_ylim(0, 110)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    for rect in rects2_1:
        h = rect.get_height()
        ax2.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in rects2_2:
        h = rect.get_height()
        ax2.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1e8449')

    # 3. VGG16 Confusion Matrix Heatmap
    ax3 = plt.subplot(2, 2, 3)
    cm_vgg = np.array([[19, 9, 2], [6, 14, 10], [2, 10, 18]])
    im1 = ax3.imshow(cm_vgg, cmap='Blues', interpolation='nearest')
    ax3.set_title('VGG16 Confusion Matrix (Acc: 56.67%)', fontsize=12, fontweight='bold')
    ax3.set_xticks([0, 1, 2])
    ax3.set_yticks([0, 1, 2])
    ax3.set_xticklabels(classes)
    ax3.set_yticklabels(classes)
    ax3.set_xlabel('Predicted Grade', fontweight='bold')
    ax3.set_ylabel('True Grade', fontweight='bold')
    plt.colorbar(im1, ax=ax3, fraction=0.046, pad=0.04)
    for i in range(3):
        for j in range(3):
            val = cm_vgg[i, j]
            color = "white" if val > 12 else "black"
            ax3.text(j, i, str(val), ha="center", va="center", color=color, fontweight='bold', fontsize=11)

    # 4. EfficientNet-B0 Confusion Matrix Heatmap
    ax4 = plt.subplot(2, 2, 4)
    cm_eff = np.array([[42, 4, 4], [1, 42, 7], [1, 5, 44]])
    im2 = ax4.imshow(cm_eff, cmap='Greens', interpolation='nearest')
    ax4.set_title('EfficientNet-B0 Confusion Matrix (Acc: 87.33%)', fontsize=12, fontweight='bold')
    ax4.set_xticks([0, 1, 2])
    ax4.set_yticks([0, 1, 2])
    ax4.set_xticklabels(classes)
    ax4.set_yticklabels(classes)
    ax4.set_xlabel('Predicted Grade', fontweight='bold')
    ax4.set_ylabel('True Grade', fontweight='bold')
    plt.colorbar(im2, ax=ax4, fraction=0.046, pad=0.04)
    for i in range(3):
        for j in range(3):
            val = cm_eff[i, j]
            color = "white" if val > 20 else "black"
            ax4.text(j, i, str(val), ha="center", va="center", color=color, fontweight='bold', fontsize=11)

    plt.suptitle("Embryo Blastocyst Grading: VGG16 vs. EfficientNet-B0 Pipeline Evaluation", 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_path = "model_comparison_chart.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated comparison figure at: {out_path}")

if __name__ == "__main__":
    generate_comparison_figure()
