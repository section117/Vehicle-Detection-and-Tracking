import numpy as np
import tensorflow as tf

# from utils import ops as utils_ops
# from utils import label_map_util
# from utils import visualization_utils as vis_util
#
# utils_ops.tf = tf.compat.v1
# tf.gfile = tf.io.gfile
# # PATH_TO_LABELS = '../bigdata/data/mscoco_label_map.pbtxt'
# PATH_TO_LABELS = './custom-data/mscoco_label_map.pbtxt'
# category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

crash_count_frames = 0

def estimate_collide(output_dict, height, width, image_np):
    global crash_count_frames
    vehicle_crash = 0
    max_curr_obj_area = 0
    centerX = centerY = 0
    details = [0, 0, 0, 0]
    for ind, scr in enumerate(output_dict['detection_classes']):
        if scr == 2 or scr == 3 or scr == 4 or scr == 6 or scr == 8:
            ymin, xmin, ymax, xmax = output_dict['detection_boxes'][ind]
            score = output_dict['detection_scores'][ind]
            if score > 0.5:
                obj_area = int((xmax - xmin) * width * (ymax - ymin) * height)
                if obj_area > max_curr_obj_area:
                    max_curr_obj_area = obj_area
                    details = [ymin, xmin, ymax, xmax]

    print(max_curr_obj_area)
    centerX, centerY = (details[1] + details[3]) / 2, (details[0] + details[2]) / 2
    if max_curr_obj_area > 70000:
        if (centerX < 0.2 and details[2] > 0.9) or (0.2 <= centerX <= 0.8) or (centerX > 0.8 and details[2] > 0.9):
            vehicle_crash = 1
            crash_count_frames = 15

    if vehicle_crash == 0:
        crash_count_frames = crash_count_frames - 1

    # cv2.putText(image_np, "{}  {}  {}  ".format(str(centerX)[:6],str(details[2])[:6],max_curr_obj_area) ,(50,100), font, 1.2,(255,255,0),2,cv2.LINE_AA)

    if crash_count_frames > 0:
        if max_curr_obj_area <= 100000:
            #cv2.putText(image_np, "YOU ARE GETTING CLOSER", (50, 50), font, 1.2, (255, 255, 0), 2, cv2.LINE_AA)
            print("YOU ARE GETTING CLOSER")
            return "YOU ARE GETTING CLOSER"
        elif max_curr_obj_area > 100000:
            #cv2.putText(image_np, "DON'T COLLIDE !!!", (50, 50), font, 1.2, (255, 255, 0), 2, cv2.LINE_AA)
            print("DON'T COLLIDE !!!")
            return "DON'T COLLIDE !!!"

    return False


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    # output_dict is a dict  with keys detection_classes , num_detections , detection_boxes(4 coordinates of each box) , detection_scores for 100 boxes
    output_dict = model(input_tensor)

    # num_detections gives number of objects in current frame
    num_detections = int(output_dict.pop('num_detections'))

    # output_dict is a dict  with keys detection_classes , detection_boxes(4 coordinates of each box) , detection_scores for num_detections boxes
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    # adding num_detections that was earlier popped out
    output_dict['num_detections'] = num_detections
    # converting all values in detection_classes as ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    # print(6,output_dict)

    return output_dict


def show_inference(model, image_path):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    # image_np = np.array(Image.open(image_path))
    image_np = np.array(image_path)
    height, width, channel = image_np.shape

    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    collision_warning = estimate_collide(output_dict, height, width, image_np)

    return collision_warning


