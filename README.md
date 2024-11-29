# Viewer Application README

## Overview

The **Viewer Application** is an OpenGL-based visualizer designed for rendering 3D objects and exploring gradient descent optimizers on data points. The tool incorporates ImGui for interactive settings, real-time rendering of optimizer trajectories, and support for multiple views (e.g., top-down, rotated, etc.). It is built to simulate, compare, and analyze various optimization algorithms.

<video controls>
  <source src="./assets/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

## Features

- **3D Visualization:**
  - Interactive trackball control for rotation, panning, and zooming.
  - Multiple camera views, including top-down and pre-defined angles.
- **Gradient Descent Simulation:**
  - Simulate trajectories of optimizers such as `SGD`, `Adam`, `RMSprop`, etc.
  - Supports adjustable learning rates and velocity settings.
- **Customizable Parameters:**
  - Adjustable optimizer settings through an intuitive GUI.
  - Dynamic selection of models and optimizers.
- **Real-time Interaction:**
  - Update visual elements on-the-fly using ImGui sliders and dropdowns.
  - Render optimizer-color legend to identify trajectories visually.

---

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/monte-carIo/CG-241.git
   cd CG-241
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy torch glfw PyOpenGL imgui sympy
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

---

## Usage Instructions

### Running the Application

- Launch the application by running `viewer.py`. The GUI will open, displaying a 3D visualization of data points and trajectories.

### Interactive Controls

1. **Mouse Controls**:
   - **Left Click & Drag**: Rotate the scene.
   - **Right Click & Drag**: Pan the scene.
   - **Scroll Wheel**: Zoom in/out.

2. **Keyboard Controls**:
   - `W`: Toggle between different fill modes (wireframe, point, solid).
   - `R`: Start/stop rotation of the scene.
   - `D`: Debug view to print projection/view matrices.
   - `0–6`: Switch between camera views.
   - `7`: Exit the camera views
   - `ESC/Q`: Exit the application.

3. **ImGui Controls**:
   - Use the dropdown menus to select models and optimizers.
   - Adjust sliders for parameters such as learning rates and velocity.

---