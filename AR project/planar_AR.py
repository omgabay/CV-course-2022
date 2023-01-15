# ======= imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
figsize = (10, 10)


# frame - image , img_to_replace ,H - Homography matrix
def create_warped_image(frame,img_to_replace,H):    
        # define the transform matrix for the *source* image in top-left,
        # top-right, bottom-right, and bottom-left order
        srcH, srcW = img_to_replace.shape[:2]       
        pts= np.float32([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]]).reshape(-1,1,2)
        dstMat = cv2.perspectiveTransform(pts, H)

        # warp the source image to the destination based on the homography
        imgH ,imgW  = frame.shape[:2]
        warped = cv2.warpPerspective(img_to_replace, H, (imgW, imgH))
        mask = np.zeros((imgH,imgW), dtype="uint8")
        cv2.fillConvexPoly(mask, dstMat.astype("int32"), (255,255,255), cv2.LINE_AA)
        # create a three channel version of the mask by stacking it depth-wise
        maskScaled = mask.copy() / 255.0
        maskScaled = np.dstack([maskScaled] * 3)

        # copy the warped source image into the input image by
        # 1) multiplying the warped image and masked together,
        # 2) then multiplying the original input image with the
        # mask (giving more weight to the input where there
        # *ARE NOT* masked pixels), and (3) adding the resulting
        # multiplications together
        warpedMultiplied = cv2.multiply(warped.astype("float"),maskScaled)
        imageMultiplied = cv2.multiply(frame.astype(float),1.0 - maskScaled)
        output = cv2.add(warpedMultiplied, imageMultiplied)
        output = output.astype("uint8")   
        return output  


def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -1)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img
def get_intrinsic_parameters():
    img_mask = "./calibration/*.jpg"
    square_size = 2.88
    pattern_size = (9,6)
    img_names = glob.glob(img_mask)

    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points = []
    plt.figure(figsize=figsize)
    
    h, w = cv2.imread(img_names[0]).shape[:2]
    for i, fn in enumerate(img_names):
        print(f'processing {fn}... ')
        imgBGR = cv2.imread(fn)

        if imgBGR is None: 
            print("Failed to load", fn)
            continue 

        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
        img  = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)

        assert w == img.shape[1] and h == img.shape[0], f"size: {img.shape[1]} X {img.shape[0]}"
        found, corners = cv2.findChessboardCorners(img,pattern_size)

        if not found: 
            print("chessboard not found")
            continue 

        if i < 12:
            img_w_corners = cv2.drawChessboardCorners(imgRGB, pattern_size, corners, found)
            plt.subplot(4, 3, i + 1)
            plt.imshow(img_w_corners)

        print(f"{fn}... OK")
        img_points.append(corners.reshape(-1,2))
        obj_points.append(pattern_points) 

    plt.show()
    # calculate camera distortion
    rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    return camera_matrix, dist_coefs      


def main():
    # ======= constants
    feature_extractor = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    # Create a FLANN matcher
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

    # === template image keypoint and descriptors
    rgb_template = cv2.cvtColor(cv2.imread("./images/monalisa.jpg"), cv2.COLOR_BGR2RGB)    
    gray_template = cv2.cvtColor(rgb_template, cv2.COLOR_RGB2GRAY)
    kp_template, desc_template = feature_extractor.detectAndCompute(gray_template, None)
    img_to_replace = cv2.imread("./images/cool_monalisa_large2.jpg")
    print(rgb_template.shape,img_to_replace.shape)

    # camera_matrix, _  = get_intrinsic_parameters()
    # print(camera_matrix)


    # ===== video input, output and metadata
    video_name = 'video_monalisa.mp4'
    cap = cv2.VideoCapture('./videos/video_monalisa.mp4')
    if not cap.isOpened():
        raise Exception(f'Could not open {video_name}')

    fps = 20
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  

    out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
    # ========== run on all frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ====== find keypoints matches of frame and template - we saw this in the SIFT notebook
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        kp_frame, desc_frame = feature_extractor.detectAndCompute(gray_frame, None)

        knn_matches = matcher.knnMatch(desc_template,desc_frame,2)
        ratio_thresh = 0.7
        good_matches = []
        good = [] 
        for match in knn_matches:
            m, n = match 
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
                good.append(match)

        good_matches.sort(key = lambda x : x.distance)    
        print(len(good_matches), frame.shape)  ##################################DELETE    

        good_match_arr = np.asarray(good)[:,0]   

        im_matches = np.empty((max(frame.shape[0], rgb_template.shape[0]), frame.shape[1]+rgb_template.shape[1], 3), dtype=np.uint8)  
        cv2.drawMatches(rgb_template, kp_template, frame, kp_frame, good_matches[:80], im_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        w, h = im_matches.shape[:2]
        im_matches = cv2.resize(im_matches, (int(0.35*w), int(0.35*h)))    
        cv2.imshow("matches template image-video" , im_matches)
        
        # ======== find homography - also in SIFT notebook :)
        good_kp_template = np.array([kp_template[m.queryIdx].pt for m in good_match_arr])
        good_kp_frame = np.array([kp_frame[m.trainIdx].pt for m in good_match_arr])

        H, _ = cv2.findHomography(good_kp_template, good_kp_frame,cv2.RANSAC,5.0)      
        output = create_warped_image(frame,img_to_replace, H)

        # display image 
        cv2.imshow("warped image", output)

         # Write the frame into the file 'output.avi'
        out.write(output)



        # ++++++++ take subset of keypoints that obey homography (both frame and reference)
        # this is at most 3 lines- 2 of which are really the same
        #kp_template_transformed = np.array(cv2.perspectiveTransform(good_kp_template, H))
        kp_template_transformed  = cv2.perspectiveTransform(good_kp_template.reshape(-1,1,2),H) 
        #print(good_kp_frame.shape, good_kp_template.shape, good_kp_template[:10], good_kp_frame[:10])
        # delta_dist = [abs(good_kp_frame[i] -good_kp_template[i]) for i in range(good_kp_frame.shape[0])] 
        # delta_dist  = np.array(delta_dist)
        # good_fit = np.argwhere(np.absolute(kp_template_transformed-kp_frame)<10)
    
        # print(good_fit, len(good_fit))
        



        # ++++++++ solve PnP to get cam pose (r_vec and t_vec)
        # `cv2.solvePnP` is a function that receives:
        # - xyz of the template in centimeter in camera world (x,3)
        # - uv coordinates (x,2) of frame that corresponds to the xyz triplets
        # - camera K
        # - camera dist_coeffs
        # and outputs the camera pose (r_vec and t_vec) such that the uv is aligned with the xyz.
        #
        # NOTICE: the first input to `cv2.solvePnP` is (x,3) vector of xyz in centimeter- but we have the template keypoints in uv
        # because they are all on the same plane we can assume z=0 and simply rescale each keypoint to the ACTUAL WORLD SIZE IN CM.
        # For this we just need the template width and height in cm.
        #
        # this part is 2 rows
        # square_size = 2.88
        # points_3D = (
        # 3
        # * square_size
        # * np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]])
        # )
        # success, rotation_vector, translation_vector = cv2.solvePnP(points_3D, points_2D, camera_matrix, dist_coeffs, flags=0)
 

        # # ++++++ draw object with r_vec and t_vec on top of rgb frame
        # # We saw how to draw cubes in camera calibration. (copy paste)
        # # after this works you can replace this with the draw function from the renderer class renderer.draw() (1 line)
        # camera_matrix, dist_coefs = get_intrinsic_parameters()
        # undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coefs)

        # =========== plot and save frame
        pass
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # ======== end all
    cap.release()
    out.release()  
    # Closes all the frames
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()





