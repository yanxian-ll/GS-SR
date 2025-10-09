import cv2
import numpy as np
import os

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_path", default="output/lower_campus_metashape/scaffold-gs/2024-12-15_232450/ortho/ours_30000/renders")
    parser.add_argument("--coordinate_path", default="output/lower_campus_metashape/scaffold-gs/2024-12-15_232450/ortho/ours_30000/coordinate")
    parser.add_argument("--output_path", default="output/lower_campus_metashape/scaffold-gs/2024-12-15_232450/ortho/ours_30000")
    args = parser.parse_args()

    image_path = args.render_path
    coord_path = args.coordinate_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    image_files = sorted([os.path.join(image_path, f) for f in os.listdir(image_path)])
    coord_files = sorted([os.path.join(coord_path, f) for f in os.listdir(coord_path)])

    list_images = []
    for f in image_files:
        image = cv2.imread(f)
        list_images.append(image)

    list_x, list_y = [], []
    for f in coord_files:
        with open(f, 'r') as fout:
            data = [float(l.strip()) for l in fout.readlines()]
            w = int(data[0])
            h = int(data[1])
            list_x.append(data[3])
            list_y.append(data[4])
            gsd = data[2]

    min_x = min(list_x)
    min_y = min(list_y)
    print(min_x, min_y, gsd)

    max_x = max(list_x)
    max_y = max(list_y)

    full_width = int((max_x - min_x) // gsd + w + 10)
    full_height = int((max_y - min_y) // gsd + h + 10)

    full_image = np.zeros((full_height, full_width, 3))
    mask = np.zeros((full_height, full_width), dtype=bool)
    temp_image = np.zeros((full_height, full_width, 3))
    temp_mask = np.zeros((full_height, full_width), dtype=bool)

    for img, x, y in zip(list_images, list_x, list_y):
        s_row = int((y-min_y)/gsd)
        s_col = int((x-min_x)/gsd)
        e_row = s_row + h
        e_col = s_col + w

        temp_image[s_row:e_row, s_col:e_col, :] = img
        temp_mask[s_row:e_row, s_col:e_col] = True

        # find intersetion
        inter_mask = np.logical_and(temp_mask, mask)
        full_image[inter_mask] = (full_image[inter_mask] + temp_image[inter_mask]) / 2.0
        false_mask = np.logical_and(temp_mask, np.logical_not(inter_mask))
        full_image[false_mask] = temp_image[false_mask]
        mask = np.logical_or(mask, temp_mask)

        temp_image = np.zeros((full_height, full_width, 3))
        temp_mask = np.zeros((full_height, full_width), dtype=bool)

    cv2.imwrite(os.path.join(output_path, 'full_image.png'), full_image)

    with open(os.path.join(output_path, 'bbx.txt'), 'w') as f:
        f.write("up_left_x up_left_y width height gsd\n")
        f.write(f"{min_x} {min_y} {full_width} {full_height} {gsd}")
