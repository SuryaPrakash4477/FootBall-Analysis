def get_centerOfbbox(bbox):
    x1, y1, x2, y2 = bbox

    return int((x1+x2)/2), int((y1+y2)/2)

def getbbox_width(bbox):
    return bbox[2]-bbox[0]

def measureDistance(p1,p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5 #equation to get the distance between two points

def measureXYDistance(p1, p2):
    return p1[0]-p2[0], p1[1]-p2[1] #distance between two y's and also between two x's

def get_FootPosition(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int(y2)
    
