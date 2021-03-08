# Fourier Based Shape Servoing Project

This is my final project for the course ECE 6141: Neural Networks for Classification and Optimization at the University of Connecticut.
In this course project, an approach for manipulating flexible objects into desired configurations using image feedback as outlined in [Navarro-Alarcon et al.](https://ieeexplore.ieee.org/document/8106734) is explored.
This approach uses a truncated Fourier series approximation of the object's 2D image contour as feedback, and minimizes the distance between the current configuration and the object's desired configuration.
This approach also models the object's deformation model as locally linear, and iteratively recalibrates this model as the object is deformed.
