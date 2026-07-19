import cv2
import numpy as np
import matplotlib.pyplot as plt


# =====================================================
# ENHANCEMENT FUNCTION
# =====================================================
def enhance_full(frame):

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    R, G, B = cv2.split(image_rgb)

    # Red enhancement
    R_enhanced = cv2.equalizeHist(R)

    # White balance
    R_mean = np.mean(R_enhanced)
    G_mean = np.mean(G)
    B_mean = np.mean(B)

    avg = (R_mean + G_mean + B_mean) / 3

    R_balanced = np.clip(R_enhanced * (avg / R_mean), 0, 255).astype(np.uint8)
    G_balanced = np.clip(G * (avg / G_mean), 0, 255).astype(np.uint8)
    B_balanced = np.clip(B * (avg / B_mean), 0, 255).astype(np.uint8)

    rgb_white = cv2.merge((R_balanced, G_balanced, B_balanced))

    # CLAHE (Final Enhancement)
    lab = cv2.cvtColor(rgb_white, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    final = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2RGB)

    return image_rgb, rgb_white, final


# =====================================================
# IMAGE PROCESSING
# =====================================================
def process_images():

    image_files = [
        "underwater img 1.jpg",
        "underwater img 2.jpg",
        "underwater img 3.webp"
    ]

    for img_path in image_files:

        image = cv2.imread(img_path)

        if image is None:
            print(f"{img_path} not found!")
            continue

        original, rgb_white, final = enhance_full(image)

        plt.figure(figsize=(12, 8))

        # Original
        plt.subplot(2, 2, 1)
        plt.imshow(original)
        plt.title("Original Image")
        plt.axis("off")

        # RGB + White Balance
        plt.subplot(2, 2, 2)
        plt.imshow(rgb_white)
        plt.title("RGB + White Balance")
        plt.axis("off")

        # Final Enhanced
        plt.subplot(2, 2, 3)
        plt.imshow(final)
        plt.title("Final Enhanced Image")
        plt.axis("off")

        # Histogram Comparison
        plt.subplot(2, 2, 4)
        plt.hist(original.ravel(), bins=256, alpha=0.5, label="Original")
        plt.hist(final.ravel(), bins=256, alpha=0.5, label="Enhanced")
        plt.title("Histogram Comparison")
        plt.xlabel("Pixel Intensity (0-255)")
        plt.ylabel("Number of Pixels")
        plt.legend()

        plt.tight_layout()
        plt.show()

        print(f"{img_path} processed successfully!")

    print("All images completed.\n")


# =====================================================
# VIDEO PROCESSING (FULLSCREEN TOP-BOTTOM)
# =====================================================
def process_videos():

    video_files = [
        "underwater vid 1.mp4",
        "underwater vid 2.mp4",
        "underwater vid 3.mp4"
    ]

    for video_path in video_files:

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"{video_path} not found!")
            continue

        print(f"Playing {video_path} (Press Q to skip)")

        cv2.namedWindow("Underwater Enhancement", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            "Underwater Enhancement",
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            original_rgb, rgb_white, final_rgb = enhance_full(frame)

            original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
            final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)

            # Resize to fit screen width
            screen_width = 1600
            h, w, _ = original_bgr.shape
            scale = screen_width / w
            new_h = int(h * scale)

            original_resized = cv2.resize(original_bgr, (screen_width, new_h))
            final_resized = cv2.resize(final_bgr, (screen_width, new_h))

            # Add Text on Original
            cv2.putText(original_resized,
                        "Original Video",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        3)

            # Add Text on Enhanced
            cv2.putText(final_resized,
                        "Enhanced Video (RGB + CLAHE)",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        3)

            # Add small gap between videos
            gap = np.zeros((30, screen_width, 3), dtype=np.uint8)

            combined = cv2.vconcat([original_resized, gap, final_resized])

            cv2.imshow("Underwater Enhancement", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    print("All videos completed.\n")


# =====================================================
# MAIN MENU
# =====================================================
if __name__ == "__main__":

    while True:

        print("\n==== UNDERWATER IMAGE ENHANCEMENT ====")
        print("1. Process Images")
        print("2. Process Videos")
        print("3. Exit")

        choice = input("Enter your choice (1/2/3): ")

        if choice == "1":
            process_images()

        elif choice == "2":
            process_videos()

        elif choice == "3":
            print("Exiting program...")
            break

        else:
            print("Invalid choice! Try again.")