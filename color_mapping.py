import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import json
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class ColorMappingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Color Mapping Tool")
        
        # Initialize variables
        # Load and resize images to half size
        self.img_a = cv2.imread('A.jpg')
        self.img_b = cv2.imread('B.jpg')
        
        # Resize images to half size
        self.img_a = cv2.resize(self.img_a, (self.img_a.shape[1]//2, self.img_a.shape[0]//2))
        self.img_b = cv2.resize(self.img_b, (self.img_b.shape[1]//2, self.img_b.shape[0]//2))
        
        self.points_a = []
        self.points_b = []
        self.current_image = 'A'
        
        # Convert BGR to RGB for display
        self.img_a_rgb = cv2.cvtColor(self.img_a, cv2.COLOR_BGR2RGB)
        self.img_b_rgb = cv2.cvtColor(self.img_b, cv2.COLOR_BGR2RGB)
        
        # Create frames
        self.create_widgets()
        
        # Bind mouse events
        self.canvas_a.bind("<Button-1>", lambda e: self.on_click(e, 'A'))
        self.canvas_b.bind("<Button-1>", lambda e: self.on_click(e, 'B'))
        
        self.update_images()
        
    def create_widgets(self):
        # Create frames for images
        self.frame_a = ttk.Frame(self.root)
        self.frame_a.grid(row=0, column=0, padx=10, pady=10)
        
        self.frame_b = ttk.Frame(self.root)
        self.frame_b.grid(row=0, column=1, padx=10, pady=10)
        
        # Create canvases for images
        self.canvas_a = tk.Canvas(self.frame_a, width=self.img_a.shape[1], height=self.img_a.shape[0])
        self.canvas_a.pack()
        
        self.canvas_b = tk.Canvas(self.frame_b, width=self.img_b.shape[1], height=self.img_b.shape[0])
        self.canvas_b.pack()
        
        # Create process button
        self.process_btn = ttk.Button(self.root, text="Process Color Mapping", command=self.process_mapping)
        self.process_btn.grid(row=1, column=0, columnspan=2, pady=10)
        
    def update_images(self):
        # Convert numpy arrays to PhotoImage
        self.photo_a = ImageTk.PhotoImage(image=Image.fromarray(self.img_a_rgb))
        self.photo_b = ImageTk.PhotoImage(image=Image.fromarray(self.img_b_rgb))
        
        # Update canvases
        self.canvas_a.create_image(0, 0, anchor="nw", image=self.photo_a)
        self.canvas_b.create_image(0, 0, anchor="nw", image=self.photo_b)
        
        # Draw points
        for x, y in self.points_a:
            self.canvas_a.create_oval(x-3, y-3, x+3, y+3, fill='red')
        
        for x, y in self.points_b:
            self.canvas_b.create_oval(x-3, y-3, x+3, y+3, fill='red')
    
    def on_click(self, event, image):
        if image == 'A':
            self.points_a.append((event.x, event.y))
        elif image == 'B':
            self.points_b.append((event.x, event.y))
        
        # Update point count display
        self.update_point_count()
        self.update_images()
    
    def update_point_count(self):
        # Clear previous count display
        for widget in self.root.grid_slaves(row=2):
            widget.destroy()
        
        # Create new count display
        count_label = ttk.Label(self.root, 
                              text=f"Points: Image A: {len(self.points_a)}, Image B: {len(self.points_b)}")
        count_label.grid(row=2, column=0, columnspan=2, pady=5)
    
    def get_rgb_values(self, img, points):
        values = []
        for x, y in points:
            # Get RGB values (normalized to 0-1)
            rgb = img[y, x] / 255.0
            values.append(rgb)
        return np.array(values)
    
    def fit_polynomial_mapping(self, X, y):
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        return poly, model
    
    def process_mapping(self):
        if len(self.points_a) != len(self.points_b):
            print(f"Please select the same number of points on both images. Currently: Image A has {len(self.points_a)} points, Image B has {len(self.points_b)} points")
            return
        
        if len(self.points_a) < 3:
            print("Please select at least 3 points on each image for a meaningful color mapping")
            return
        
        # Get RGB values for selected points
        rgb_values_a = self.get_rgb_values(self.img_a, self.points_a)
        rgb_values_b = self.get_rgb_values(self.img_b, self.points_b)
        
        # Fit polynomial mapping for each channel
        coefficients = {}
        transformed_image = np.zeros_like(self.img_a, dtype=np.float32)
        
        for i, channel in enumerate(['B', 'G', 'R']):
            X = rgb_values_a[:, i].reshape(-1, 1)
            y = rgb_values_b[:, i]
            
            poly, model = self.fit_polynomial_mapping(X, y)
            
            # Store coefficients
            coefficients[channel] = {
                'intercept': float(model.intercept_),
                'coefficients': model.coef_.tolist()
            }
            
            # Transform the entire channel
            channel_data = self.img_a[:, :, i].reshape(-1, 1) / 255.0
            channel_data_poly = poly.transform(channel_data)
            transformed_channel = model.predict(channel_data_poly)
            transformed_image[:, :, i] = transformed_channel.reshape(self.img_a.shape[0], self.img_a.shape[1])
        
        # Clip values to valid range and convert to uint8
        transformed_image = np.clip(transformed_image * 255, 0, 255).astype(np.uint8)
        
        # Save results
        cv2.imwrite('transformed_A.png', transformed_image)
        with open('mapping_coefficients.json', 'w') as f:
            json.dump(coefficients, f, indent=4)
        
        print("Results saved as 'transformed_A.png' and 'mapping_coefficients.json'")

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorMappingApp(root)
    root.mainloop() 