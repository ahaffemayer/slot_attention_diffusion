---
layout: default
title: Collision-Free Model Predictive Control with Diffusion Model Warm-Starting

---

<div style="max-width:720px; margin:16px auto;">
  <video
    controls
    autoplay
    loop
    muted
    playsinline
    preload="metadata"
    style="width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15); background:#000;"
  >
    <source src="{{ '/static/RAL2025.mp4' | relative_url }}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>



# Abstract

Dynamic motion planning for complex robots in cluttered environments is essential for enabling robots to perform manipulation tasks in dynamic real-world settings. However, generating collision-free and dynamically feasible trajectories remains challenging due to the complexities of often non-convex environments and stringent constraints. To address this, we propose a novel hybrid framework that combines the strengths of data-driven diffusion models with model-based optimal control. Our approach leverages a diffusion model, conditioned on object-centric scene representations, to generate robust and generalizable trajectory priors across diverse environments. These learned priors are then refined within a Model Predictive Control (MPC) loop. This integrated approach ensures collision avoidance, dynamic feasibility, and improved performance by effectively blending learned trajectory generation with model-based planning. Experiments, conducted in simulation and on a real 7-DoF manipulator, demonstrate that our approach significantly improves success rates and constraint satisfaction in demanding, real-world settings, highlighting its effectiveness in dynamic motion planning.

# Method

<div style="text-align:center;">
  <img src="static/pipeline.png" width="720" alt="Architecture diagram" style="border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.2);">
  <p style="font-size:14px; color:gray; margin-top:5px;">
    Figure 1: Overview of our proposed architecture.
  </p>
</div>

Our pipeline first encodes the scene and task, using Slot-Attention features from a rendered view plus the current state and goal, to build a compact conditioning vector. A conditional diffusion model denoises a noise sequence into a full trajectory prior, giving a warm-start that is close to feasible. This trajectory initializes an MPC solver that enforces robot dynamics and hard collision constraints, refining the motion to be executable. The resulting controls are applied on the robot, feedback updates the state, and the loop repeats.

# Results 

The benchmark environments are divided into three difficulty levels based on the number and placement of obstacles. Level 1 features a nearly free workspace with only a single randomly placed obstacle, making motion planning relatively easy. Level 2 introduces moderate clutter by placing one obstacle on each shelf level, increasing the need for careful collision avoidance. Level 3 represents the most challenging setting, with up to two obstacles per shelf level, resulting in highly constrained, non-convex environments that require precise and robust planning strategies.

<div style="display: flex; justify-content: center; gap: 15px; flex-wrap: wrap; margin-top: 10px;">

  <video autoplay loop muted playsinline width="320" style="border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
    <source src="static/scene_1.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video autoplay loop muted playsinline width="320" style="border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
    <source src="static/scene_2.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video autoplay loop muted playsinline width="320" style="border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
    <source src="static/scene_3.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

</div>



## Comparison of the different conditioning methods

<div style="text-align:center;">
  <img src="static/condition_comparison.png" width="720" alt="Conditioning comparison" style="border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.2);">
  <p style="font-size:14px; color:gray; margin-top:5px;">
    Figure 2: Comparing the different conditioning methods.
  </p>
</div>

Among the conditioning options, Slot Attention works best overall, giving the highest success at low to mid compute budgets and fast inference thanks to its compact object centric tokens. Occupancy grids come close but are heavier to build and slower at runtime, which hurts when you need quick replans. C space encodings from key configurations help in simple scenes, but they generalize less and require precomputing collision labels. Pure image based conditioning performs worst, it saturates early and misses geometric detail needed for reliable collision avoidance.


## Motion generation methods comparison

<div style="text-align:center;">
  <img src="static/comparison_sota_with_cost.png" width="720" alt="SOTA comparison" style="border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.2);">
  <p style="font-size:14px; color:gray; margin-top:5px;">
    Figure 3: Comparing the different motion generation methods.
  </p>
</div>

Compared to existing baselines, our approach consistently achieves the highest success rates across all difficulty levels while maintaining low computation times. RRT performs well in simple environments but struggles in cluttered scenes and is significantly slower, making it unsuitable for real-time control. A pure OCP solver without warm-starts fails in most cases due to sensitivity to local minima. Motion Planning Diffusion (MPD) improves over OCP but lacks the dynamic feasibility guarantees provided by our integrated refinement. Overall, combining diffusion priors with OCP refinement yields smoother, safer, and more efficient trajectories, outperforming both classical and learning-based alternatives.

## Experiments on a real robot.

The experiments were done on a Franka Emika Panda manipulator.

### Exploring the workspace
<div style="display: flex; justify-content: center; gap: 15px; flex-wrap: wrap; margin-top: 10px;">

  <video autoplay loop muted playsinline width="320" style="border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
    <source src="static/demo1.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video autoplay loop muted playsinline width="320" style="border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
    <source src="static/demo2.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video autoplay loop muted playsinline width="320" style="border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
    <source src="static/demo3.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

</div>

### Adapting the trajectory to avoid collisions

<div style="display: flex; justify-content: center; gap: 15px; flex-wrap: wrap; margin-top: 10px;">

  <video autoplay loop muted playsinline width="320" style="border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
    <source src="static/scene_1_to_1_final.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video autoplay loop muted playsinline width="320" style="border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
    <source src="static/scene_3_final.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  </div>
