import os
import cv2
from pathlib import Path
import numpy as np
import shutil

class_names = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B0",
    "BS",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R0",
    "RS",
]

LABELS_PATH = "/home/fallengold/extra/armor_opensource/XJTLU_2023_Detection_ALL/labels"
IMAGES_PATH = "/home/fallengold/extra/armor_opensource/XJTLU_2023_Detection_ALL/images"
OUTPUT_PATH = "/home/fallengold/extra/pure_armor_dataset"


class ArmorDigitDatasetBuilder:

    def __init__(self, labels_path, images_path, output_path):
        self.labels_path = Path(labels_path)
        self.images_path = Path(images_path)
        self.output_path = Path(output_path)

        # Create output directory structure
        self.output_path.mkdir(exist_ok=True)
        (self.output_path / "images").mkdir(exist_ok=True)
        (self.output_path / "labels").mkdir(exist_ok=True)

        self.crop_count = 0
        self.error_count = 0

    def load_annotations(self, label_file):
        """Load all annotations from a label file"""
        annotations = []
        label_path = self.labels_path / label_file

        try:
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines):
                try:
                    line = line.strip()
                    if not line:
                        continue

                    split = line.split()
                    if len(split) < 5:
                        print(
                            f"Warning: Line {line_num + 1} in {label_file} has insufficient data: {len(split)} items"
                        )
                        continue

                    class_id = int(split[0])

                    # Validate class_id
                    if class_id < 0 or class_id >= len(class_names):
                        print(
                            f"Warning: Invalid class_id {class_id} in {label_file}, line {line_num + 1}"
                        )
                        continue

                    # Extract bounding box (xc, yc, w, h)
                    xc, yc, w, h = map(float, split[1:5])

                    # Validate bounding box values
                    if not all(0 <= val <= 1 for val in [xc, yc, w, h]):
                        print(
                            f"Warning: Invalid bbox values in {label_file}, line {line_num + 1}: {[xc, yc, w, h]}"
                        )
                        continue

                    annotations.append((class_id, [xc, yc, w, h]))

                except (ValueError, IndexError) as e:
                    print(f"Error parsing line {line_num + 1} in {label_file}: {e}")
                    self.error_count += 1
                    continue

        except FileNotFoundError:
            print(f"Error: Label file {label_file} not found")
            return []
        except Exception as e:
            print(f"Error reading {label_file}: {e}")
            return []

        return annotations

    def denormalize_bbox(self, bbox, img_height, img_width):
        """Convert normalized bbox to pixel coordinates"""
        xc, yc, w, h = bbox

        # Convert to pixel coordinates
        xc_pixel = xc * img_width
        yc_pixel = yc * img_height
        w_pixel = w * img_width
        h_pixel = h * img_height

        # Convert to x1, y1, x2, y2
        x1 = int(xc_pixel - w_pixel / 2)
        y1 = int(yc_pixel - h_pixel / 2)
        x2 = int(xc_pixel + w_pixel / 2)
        y2 = int(yc_pixel + h_pixel / 2)

        # Clamp to image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)

        return x1, y1, x2, y2

    def crop_and_save(self, image, bbox, class_id, image_name, annotation_idx):
        """Crop image region and save with label"""
        x1, y1, x2, y2 = bbox

        # Validate crop region
        if x2 <= x1 or y2 <= y1:
            print(
                f"Warning: Invalid crop region for {image_name}, annotation {annotation_idx}"
            )
            return False

        # Crop image
        cropped = image[y1:y2, x1:x2]

        if cropped.size == 0:
            print(f"Warning: Empty crop for {image_name}, annotation {annotation_idx}")
            return False

        # Generate output filename
        base_name = Path(image_name).stem
        output_name = f"{base_name}_{annotation_idx:03d}"

        # Save cropped image
        image_output_path = self.output_path / "images" / f"{output_name}.jpg"
        cv2.imwrite(str(image_output_path), cropped)

        # Save label
        label_output_path = self.output_path / "labels" / f"{output_name}.txt"
        with open(label_output_path, "w") as f:
            f.write(f"{class_id}\n")

        self.crop_count += 1
        return True

    def process_single_image(self, image_file):
        """Process a single image and its corresponding label file"""
        # Get corresponding label file
        label_file = Path(image_file).stem + ".txt"

        # Load image
        image_path = self.images_path / image_file
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Error: Could not load image {image_file}")
                return 0

            img_height, img_width = image.shape[:2]

        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
            return 0

        # Load annotations
        annotations = self.load_annotations(label_file)
        if not annotations:
            print(f"No valid annotations found for {image_file}")
            return 0

        # Process each annotation
        processed_count = 0
        for idx, (class_id, bbox) in enumerate(annotations):
            try:
                # Denormalize bbox
                x1, y1, x2, y2 = self.denormalize_bbox(bbox, img_height, img_width)

                # Crop and save
                if self.crop_and_save(
                    image, (x1, y1, x2, y2), class_id, image_file, idx
                ):
                    processed_count += 1

            except Exception as e:
                print(f"Error processing annotation {idx} for {image_file}: {e}")
                self.error_count += 1
                continue

        return processed_count

    def build_dataset(self):
        """Build the complete dataset"""
        print("Starting dataset construction...")
        print(f"Input images path: {self.images_path}")
        print(f"Input labels path: {self.labels_path}")
        print(f"Output path: {self.output_path}")

        # Get all image files
        try:
            image_files = [
                f
                for f in os.listdir(self.images_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]
            image_files.sort()

        except Exception as e:
            print(f"Error accessing images directory: {e}")
            return

        if not image_files:
            print("No image files found!")
            return

        print(f"Found {len(image_files)} image files")

        # Process each image
        total_crops = 0
        for i, image_file in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_file}")
            crops_from_image = self.process_single_image(image_file)
            total_crops += crops_from_image

            if (i + 1) % 100 == 0:
                print(
                    f"Progress: {i+1}/{len(image_files)} images processed, {total_crops} crops saved"
                )

        # Print summary
        print("\n" + "=" * 50)
        print("Dataset construction completed!")
        print(f"Total images processed: {len(image_files)}")
        print(f"Total crops saved: {total_crops}")
        print(f"Total errors encountered: {self.error_count}")
        print(f"Output directory: {self.output_path}")

        # Print class distribution
        self.print_class_distribution()

    def print_class_distribution(self):
        """Print distribution of classes in the dataset"""
        print("\nClass distribution:")

        label_files = list((self.output_path / "labels").glob("*.txt"))
        class_counts = {i: 0 for i in range(len(class_names))}

        for label_file in label_files:
            try:
                with open(label_file, "r") as f:
                    class_id = int(f.read().strip())
                    if class_id in class_counts:
                        class_counts[class_id] += 1
            except:
                continue

        for class_id, count in class_counts.items():
            if count > 0:
                print(f"  {class_names[class_id]}: {count}")

    def visualize_samples(
        self, class_name=None, num_samples=10, resize_factor=3, target_size=(64, 64)
    ):
        """
        Visualize random samples from the dataset

        Args:
            class_name: Name of the class to visualize (e.g., "B1", "R2"). If None, show all classes
            num_samples: Number of samples to show per class
            resize_factor: Factor to resize images for better visibility
            target_size: Target size (width, height) to resize all images to before scaling
        """
        import random

        # Get all label files
        label_files = list((self.output_path / "labels").glob("*.txt"))

        if not label_files:
            print("No label files found in the dataset!")
            return

        # Group files by class
        class_files = {name: [] for name in class_names}

        for label_file in label_files:
            try:
                with open(label_file, "r") as f:
                    class_id = int(f.read().strip())
                    if 0 <= class_id < len(class_names):
                        # Get corresponding image file
                        image_file = (
                            self.output_path / "images" / f"{label_file.stem}.jpg"
                        )
                        if image_file.exists():
                            class_files[class_names[class_id]].append(image_file)
            except:
                continue

        # Filter by specific class if requested
        if class_name:
            if class_name not in class_names:
                print(f"Error: Class '{class_name}' not found in class_names")
                return
            classes_to_show = {class_name: class_files[class_name]}
        else:
            classes_to_show = {
                name: files for name, files in class_files.items() if files
            }

        # Visualize samples
        for cls_name, files in classes_to_show.items():
            if not files:
                print(f"No samples found for class '{cls_name}'")
                continue

            print(f"\nVisualizing class '{cls_name}' ({len(files)} total samples)")

            # Sample random files
            sample_files = random.sample(files, min(num_samples, len(files)))

            # Load and display images
            images_to_show = []
            valid_samples = 0

            for img_file in sample_files:
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        # First resize to target size for uniformity
                        img_uniform = cv2.resize(
                            img, target_size, interpolation=cv2.INTER_LINEAR
                        )

                        # Then scale up for better visibility
                        new_h, new_w = (
                            target_size[1] * resize_factor,
                            target_size[0] * resize_factor,
                        )
                        img_resized = cv2.resize(
                            img_uniform, (new_w, new_h), interpolation=cv2.INTER_NEAREST
                        )

                        # Add filename as title
                        img_with_text = img_resized.copy()
                        cv2.putText(
                            img_with_text,
                            img_file.stem,
                            (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 255, 0),
                            1,
                        )

                        images_to_show.append(img_with_text)
                        valid_samples += 1
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
                    continue

            if not images_to_show:
                print(f"No valid images found for class '{cls_name}'")
                continue

            # Arrange images in a grid
            grid_img = self._create_image_grid(images_to_show, cls_name)

            # Display the grid
            window_name = f"Class {cls_name} - {valid_samples} samples"
            cv2.imshow(window_name, grid_img)

            print(
                f"Showing {valid_samples} samples for class '{cls_name}'. Press any key to continue..."
            )
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)

    def _create_image_grid(self, images, class_name, max_cols=5):
        """Create a grid layout for multiple images"""
        if not images:
            return None

        # Calculate grid dimensions
        num_images = len(images)
        cols = min(max_cols, num_images)
        rows = (num_images + cols - 1) // cols

        # Find the maximum dimensions among all images
        max_h = max(img.shape[0] for img in images)
        max_w = max(img.shape[1] for img in images)

        # Create grid canvas
        grid_h = rows * max_h + (rows - 1) * 10  # 10px spacing
        grid_w = cols * max_w + (cols - 1) * 10  # 10px spacing
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        # Place images in grid
        for i, img in enumerate(images):
            row = i // cols
            col = i % cols

            # Calculate position for this cell
            y_start = row * (max_h + 10)
            x_start = col * (max_w + 10)

            # Get actual image dimensions
            img_h, img_w = img.shape[:2]

            # Center the image in the cell if it's smaller than max dimensions
            y_offset = (max_h - img_h) // 2
            x_offset = (max_w - img_w) // 2

            # Place the image
            grid[
                y_start + y_offset : y_start + y_offset + img_h,
                x_start + x_offset : x_start + x_offset + img_w,
            ] = img

        # Add class name as title
        title_height = 30
        title_canvas = np.zeros((title_height, grid_w, 3), dtype=np.uint8)
        cv2.putText(
            title_canvas,
            f"Class: {class_name}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Combine title and grid
        final_grid = np.vstack([title_canvas, grid])

        return final_grid

    def show_class_statistics(self):
        """Show detailed statistics about the dataset"""
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)

        label_files = list((self.output_path / "labels").glob("*.txt"))
        class_counts = {i: 0 for i in range(len(class_names))}

        for label_file in label_files:
            try:
                with open(label_file, "r") as f:
                    class_id = int(f.read().strip())
                    if class_id in class_counts:
                        class_counts[class_id] += 1
            except:
                continue

        total_samples = sum(class_counts.values())

        print(f"Total samples: {total_samples}")
        print(
            f"Number of classes with data: {sum(1 for count in class_counts.values() if count > 0)}"
        )
        print("\nClass distribution:")
        print("-" * 40)

        for class_id, count in class_counts.items():
            if count > 0:
                percentage = (count / total_samples) * 100 if total_samples > 0 else 0
                print(
                    f"  {class_names[class_id]:>3}: {count:>6} samples ({percentage:>5.1f}%)"
                )

        # Find classes with no data
        empty_classes = [
            class_names[i] for i, count in class_counts.items() if count == 0
        ]
        if empty_classes:
            print(f"\nClasses with no data: {', '.join(empty_classes)}")

    def split_train_val(self, val_ratio=0.2, random_seed=42):
        """
        Split the dataset into train and validation sets

        Args:
            val_ratio: Ratio of validation set (0.0 to 1.0)
            random_seed: Random seed for reproducible splits
        """
        import random
        import shutil

        random.seed(random_seed)

        # Create train/val directory structure
        train_path = self.output_path / "train"
        val_path = self.output_path / "val"

        for split_path in [train_path, val_path]:
            split_path.mkdir(exist_ok=True)
            (split_path / "images").mkdir(exist_ok=True)
            (split_path / "labels").mkdir(exist_ok=True)

        # Get all label files
        label_files = list((self.output_path / "labels").glob("*.txt"))

        if not label_files:
            print("No label files found for splitting!")
            return

        # Group files by class for stratified split
        class_files = {name: [] for name in class_names}

        for label_file in label_files:
            try:
                with open(label_file, "r") as f:
                    class_id = int(f.read().strip())
                    if 0 <= class_id < len(class_names):
                        class_files[class_names[class_id]].append(label_file.stem)
            except:
                continue

        print(f"Splitting dataset with validation ratio: {val_ratio}")
        print("Class-wise split:")
        print("-" * 50)

        total_train = 0
        total_val = 0

        # Split each class separately (stratified split)
        for class_name, file_stems in class_files.items():
            if not file_stems:
                continue

            # Shuffle files for this class
            random.shuffle(file_stems)

            # Calculate split point
            num_files = len(file_stems)
            num_val = int(num_files * val_ratio)
            num_train = num_files - num_val

            # Split files
            train_files = file_stems[:num_train]
            val_files = file_stems[num_train:]

            print(
                f"  {class_name}: {num_train} train, {num_val} val (total: {num_files})"
            )

            # Move files to train set
            for file_stem in train_files:
                self._move_file_to_split(file_stem, train_path)
                total_train += 1

            # Move files to val set
            for file_stem in val_files:
                self._move_file_to_split(file_stem, val_path)
                total_val += 1

        print("-" * 50)
        print(f"Total: {total_train} train, {total_val} val samples")
        print(f"Train/Val split completed successfully!")
        print(f"Train data: {train_path}")
        print(f"Val data: {val_path}")

        # Clean up original images and labels folders
        try:
            shutil.rmtree(self.output_path / "images")
            shutil.rmtree(self.output_path / "labels")
            print("Cleaned up original images and labels folders")
        except:
            print("Warning: Could not clean up original folders")

    def _move_file_to_split(self, file_stem, split_path):
        """Move image and label files to train/val split"""
        # Move image file
        src_img = self.output_path / "images" / f"{file_stem}.jpg"
        dst_img = split_path / "images" / f"{file_stem}.jpg"
        if src_img.exists():
            shutil.move(str(src_img), str(dst_img))

        # Move label file
        src_label = self.output_path / "labels" / f"{file_stem}.txt"
        dst_label = split_path / "labels" / f"{file_stem}.txt"
        if src_label.exists():
            shutil.move(str(src_label), str(dst_label))

    def create_pytorch_dataset_structure(self, val_ratio=0.2, random_seed=42):
        """
        Create PyTorch-style dataset structure with class folders

        Creates structure like:
        dataset/
        ├── train/
        │   ├── B1/
        │   ├── B2/
        │   └── ...
        └── val/
            ├── B1/
            ├── B2/
            └── ...
        """
        import random
        import shutil

        random.seed(random_seed)

        # Create PyTorch-style directory structure
        pytorch_path = self.output_path / "pytorch_format"
        train_path = pytorch_path / "train"
        val_path = pytorch_path / "val"

        # Create class folders
        for split_path in [train_path, val_path]:
            split_path.mkdir(parents=True, exist_ok=True)
            for class_name in class_names:
                (split_path / class_name).mkdir(exist_ok=True)

        # Get all label files
        label_files = list((self.output_path / "labels").glob("*.txt"))

        if not label_files:
            print("No label files found for PyTorch dataset creation!")
            return

        # Group files by class
        class_files = {name: [] for name in class_names}

        for label_file in label_files:
            try:
                with open(label_file, "r") as f:
                    class_id = int(f.read().strip())
                    if 0 <= class_id < len(class_names):
                        class_name = class_names[class_id]
                        image_file = (
                            self.output_path / "images" / f"{label_file.stem}.jpg"
                        )
                        if image_file.exists():
                            class_files[class_name].append(image_file)
            except:
                continue

        print(f"Creating PyTorch dataset structure with validation ratio: {val_ratio}")
        print("Class-wise split:")
        print("-" * 50)

        total_train = 0
        total_val = 0

        # Split each class separately
        for class_name, image_files in class_files.items():
            if not image_files:
                continue

            # Shuffle files for this class
            random.shuffle(image_files)

            # Calculate split point
            num_files = len(image_files)
            num_val = int(num_files * val_ratio)
            num_train = num_files - num_val

            # Split files
            train_files = image_files[:num_train]
            val_files = image_files[num_train:]

            print(
                f"  {class_name}: {num_train} train, {num_val} val (total: {num_files})"
            )

            # Copy files to train set
            for img_file in train_files:
                dst = train_path / class_name / img_file.name
                shutil.copy2(str(img_file), str(dst))
                total_train += 1

            # Copy files to val set
            for img_file in val_files:
                dst = val_path / class_name / img_file.name
                shutil.copy2(str(img_file), str(dst))
                total_val += 1

        print("-" * 50)
        print(f"Total: {total_train} train, {total_val} val samples")
        print(f"PyTorch dataset structure created at: {pytorch_path}")

        return pytorch_path

    def show_split_statistics(self, split_path=None):
        """Show statistics for train/val split"""
        if split_path is None:
            train_path = self.output_path / "train"
            val_path = self.output_path / "val"
        else:
            train_path = split_path / "train"
            val_path = split_path / "val"

        print("\n" + "=" * 60)
        print("TRAIN/VAL SPLIT STATISTICS")
        print("=" * 60)

        for split_name, path in [("TRAIN", train_path), ("VAL", val_path)]:
            print(f"\n{split_name} SET:")
            print("-" * 30)

            if not path.exists():
                print(f"  Path does not exist: {path}")
                continue

            # Check if it's PyTorch format (class folders) or YOLO format (images/labels)
            if (path / "images").exists():
                # YOLO format
                label_files = list((path / "labels").glob("*.txt"))
                class_counts = {i: 0 for i in range(len(class_names))}

                for label_file in label_files:
                    try:
                        with open(label_file, "r") as f:
                            class_id = int(f.read().strip())
                            if class_id in class_counts:
                                class_counts[class_id] += 1
                    except:
                        continue

                total = sum(class_counts.values())
                print(f"  Total samples: {total}")

                for class_id, count in class_counts.items():
                    if count > 0:
                        percentage = (count / total) * 100 if total > 0 else 0
                        print(
                            f"    {class_names[class_id]}: {count:>4} ({percentage:>5.1f}%)"
                        )

            else:
                # PyTorch format
                total = 0
                for class_name in class_names:
                    class_path = path / class_name
                    if class_path.exists():
                        count = len(list(class_path.glob("*.jpg")))
                        if count > 0:
                            total += count

                print(f"  Total samples: {total}")

                for class_name in class_names:
                    class_path = path / class_name
                    if class_path.exists():
                        count = len(list(class_path.glob("*.jpg")))
                        if count > 0:
                            percentage = (count / total) * 100 if total > 0 else 0
                            print(f"    {class_name}: {count:>4} ({percentage:>5.1f}%)")


def main():
    # Build the dataset
    builder = ArmorDigitDatasetBuilder(LABELS_PATH, IMAGES_PATH, OUTPUT_PATH)
    # builder.build_dataset()
    # builder.show_class_statistics(
    builder.visualize_samples(class_name="BS", num_samples=30, resize_factor=3)
    # builder.create_pytorch_dataset_structure(val_ratio=0.1)


if __name__ == "__main__":
    main()
