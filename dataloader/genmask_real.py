class GenMask_real(object):
    def __init__(self):
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1, 30)]

    def __call__(self, sample):
        
        alpha_ori = sample['alpha']
        h, w = alpha_ori.shape

        # Perform connected component analysis
        _, labels, stats, centroids = cv2.connectedComponentsWithStats((alpha_ori > 0).astype(np.uint8), connectivity=8)

        
        # Exclude the background (label 0)
        num_components = len(np.unique(labels)) - 1
        valid_components = []

        for i in range(1, num_components + 1):
            # Get the area of each component (stat[cv2.CC_STAT_AREA])
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 500:
                valid_components.append(i)
        
        if valid_components:
            # Randomly select one of the valid components (area > 500)
            chosen_component = np.random.choice(valid_components)
            
            # Create a mask that only contains the chosen component
            component_mask = (labels == chosen_component).astype(np.uint8)
        else:
            # If no valid component, no connected component analysis is done
            component_mask = np.ones_like(alpha_ori, dtype=np.uint8)  # No modification, keep original alpha_ori

        # Apply the component mask to alpha_ori
        alpha_ori = alpha_ori * component_mask

        max_kernel_size = 15  # 30
        alpha = cv2.resize(alpha_ori, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

        fg_width = np.random.randint(1, max_kernel_size)
        bg_width = np.random.randint(1, max_kernel_size)
        fg_mask = (alpha + 1e-5).astype(int).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(int).astype(np.uint8)
        fg_mask = cv2.erode(fg_mask, self.erosion_kernels[fg_width])
        bg_mask = cv2.erode(bg_mask, self.erosion_kernels[bg_width])

        trimap = np.ones_like(alpha) * 128
        trimap[fg_mask == 1] = 255
        trimap[bg_mask == 1] = 0

        trimap = cv2.resize(trimap, (w, h), interpolation=cv2.INTER_NEAREST)
        sample['trimap'] = trimap

        # Generate mask for segmentation
        seg_mask = (alpha >= 0.01).astype(int).astype(np.uint8)
        seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        sample['mask'] = seg_mask
        sample['alpha'] = alpha_ori

        return sample
