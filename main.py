import cv2, numpy as np, os

def square_extractor(img):
    # img=cv2.imread(f'imgs/{file}')
    square=img[:,:,1].copy()
    ### Histogram normalization
    square=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(square)
    ### Binary Image Thresholding
    square=cv2.threshold(square,-1,255,cv2.THRESH_OTSU|cv2.THRESH_BINARY)[-1]
    ### Connected components
    ### cc_box => cover box of a connected component : x, y, width, height, area
    cc_count, cc_map, cc_box, cc_center=cv2.connectedComponentsWithStats(square)
    id_=np.argmax(cc_box[1:,-1])+1
    width=np.max(cc_box[id_][2:4])
    square=cv2.morphologyEx(
        square,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (width//30,width//30)),
    )
    ### cc_map=> number of cover boxes of connected components
    cc_count, cc_map, cc_box, cc_center=cv2.connectedComponentsWithStats(square)
    id_=np.argmax(cc_box[1:,-1])+1
    square=np.where(cc_map==id_, 255, 0).astype('uint8')
    width=np.max(cc_box[id_][2:4])
    square=cv2.morphologyEx(
        square,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (width//10,width//10)),
    )
    square=cv2.morphologyEx(
        square,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (width//10,width//10)),
    )
    return square

def clear2binary(img, M=230.74238275948176):
    square=square_extractor(img)

    ''''''''' find width of image '''''''''
    min_y,max_y,min_x,max_x=(lambda i: (
        np.min(i[0]),np.max(i[0]),
        np.min(i[1]),np.max(i[1]),
    ))(np.where(square==255))
    width=np.mean([max_y-min_y+1,max_x-min_x+1]).astype('int')
    
    ''''''''' apply adaptive functions in square '''''''''
    gray=img[:,:,1].copy()
    offset=M-np.mean(gray[square==255])
    gray=np.clip(gray.astype('float32')+offset,0,255).astype('uint8')
    gray[min_y:max_y+1,min_x:max_x+1]=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(
        gray[min_y:max_y+1,min_x:max_x+1]
    )
    
    ''''''''' shadow issue '''''''''
    shadow=gray[min_y:max_y+1,min_x:max_x+1].copy()
    shadow=cv2.morphologyEx(
        shadow,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (width//10,width//10)),
    )
    r=int(width/3)
    r=r if r%2!=0 else r-1
    r=r if r<=255 else 255
    shadow=cv2.medianBlur(shadow, r)
    shadow=255-shadow
    
    gray[min_y:max_y+1,min_x:max_x+1]=np.clip(
        gray[min_y:max_y+1,min_x:max_x+1].copy().astype('int')+shadow.astype('int'),
        0, 255
    ).astype('uint8')
    
    gray=cv2.threshold(
        gray,-1,255,
        cv2.THRESH_OTSU|cv2.THRESH_BINARY
    )[-1]
    
    ''''''''' clear outter '''''''''
    gray[square==0]=255
    
    ''''''''' repair circle '''''''''
    gray=cv2.morphologyEx(
        gray,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width//20,width//20)),
    )
    
    ''''''''' clear weak lines '''''''''
    gray=cv2.morphologyEx(
        gray,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width//25,width//25)),
    )
    return gray, (square, [min_y,max_y,min_x,max_x])

def find_dots(img, iou_t=0.6, whr_t=0.5):
    binary, (_, [min_y,max_y,min_x,max_x]) = clear2binary(img.copy())
    cc_count, cc_map, cc_box, cc_center=cv2.connectedComponentsWithStats(255-binary)
    boxes=cc_box[1:]
    ious=boxes[:,-1]/(boxes[:,2]*boxes[:,3])
    whr=np.abs((boxes[:,2]/boxes[:,3])-1.0)
    boxes=boxes[(ious>iou_t)&(whr<whr_t)]
    ''''''''' top 4 area '''''''''
    boxes=boxes[np.argsort(boxes[:,-1])[::-1]]
    boxes=boxes[:4]
    return boxes

def sort_dots(dot_locations):
    dot_locations=np.array(sorted(dot_locations, key=lambda x : x[0]))
    dot_locations[:2]=np.array(sorted(dot_locations[:2], key=lambda x : x[1]))
    dot_locations[2:]=np.array(sorted(dot_locations[2:], key=lambda x : x[1]))
    return dot_locations

if __name__=='__main__':
    for file in os.listdir('imgs'):
        ### Load image
        test_board=cv2.imread(f'imgs/{file}')
        ### Load Line template
        template_board=np.load('template.npy').astype('uint8')
        
        ### Find 4 dots and sort
        dot_locations=find_dots(test_board.copy())
        assert(len(dot_locations)==4), 'fail'
        dot_locations=dot_locations[:,:2]+(dot_locations[:,2:4]/2)
        dot_locations=sort_dots(dot_locations)
        
        ### Rotate image if need
        if np.linalg.norm(dot_locations[1]-dot_locations[0])<np.linalg.norm(dot_locations[3]-dot_locations[2]):
            test_board=cv2.rotate(test_board, cv2.ROTATE_180)
            dot_locations[:,0]=test_board.shape[1]-dot_locations[:,0]
            dot_locations[:,1]=test_board.shape[0]-dot_locations[:,1]
            dot_locations=sort_dots(dot_locations)
        
        ### Find Affine matrix and transform image to fit template
        template_board=cv2.warpAffine(
            template_board,
            cv2.getAffineTransform(
                np.float32(np.array([
                    np.concatenate(np.where(template_board==10)[::-1], axis=0),
                    np.concatenate(np.where(template_board==11)[::-1], axis=0),
                    np.concatenate(np.where(template_board==12)[::-1], axis=0)
                ])),
                np.float32(dot_locations[:3])
            ),
            test_board.shape[:2][::-1],
            flags=cv2.INTER_NEAREST
        )
        template_board[template_board>9]=0 ####################
        d_rate=int(np.max(template_board.shape[:2])/100)

        ### Dilate line mask for fault tolerance of area comparison
        template_board=cv2.dilate(
            template_board,
            cv2.getStructuringElement(cv2.MORPH_RECT, (d_rate,d_rate))
        )
        
        ### Evaluate Each lines average value and output it rbg values respectively
        print('-'*5, file, '-'*5)
        for wire_id in range(1, 10):
            wire=test_board[np.where(template_board==wire_id)]
            maxb, maxg, maxr=np.max(wire[:,0]), np.max(wire[:,1]), np.max(wire[:,2])
            minb, ming, minr=np.min(wire[:,0]), np.min(wire[:,1]), np.min(wire[:,2])
            avgb, avgg, avgr=np.average(wire[:,0]), np.average(wire[:,1]), np.average(wire[:,2])
            print(f'Line {wire_id}: max_b:{maxb}, max_g:{maxg}, max_r:{maxr}')
            print(f'Line {wire_id}: min_b:{minb}, min_g:{ming}, min_r:{minr}')
            print(f'Line {wire_id}: avg_b:{avgb}, avg_g:{avgg}, avg_r:{avgr}')






