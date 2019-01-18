import numpy as np
import matplotlib.path as mplot_path


def shape_to_np(shape, dtype="int"):
    """
    Convert a landmark shape object to a numpy array of coordinates
    :param shape: original shape
    :param dtype: datatype of the resulting coordinatis (default int)
    :return: numpy array ofcoordinates (x,y)
    """
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def create_blank_image(width, height, rgb_color=(0, 0, 0)):
    """
    Calculate f(x) = ax + b for the specified punts
    :param width: Width of the image to create
    :param height: Height of the image to create
    :param rgb_color: Background color, default black
    :return image
    """
    image = np.zeros((height, width, 3), np.uint8)
    color = tuple(reversed(rgb_color))
    image[:] = color
    return image


def rect_contains(rect, point):
    """
    Check if a point is inside the rectangle
    :param rect: rectangle
    :param  point: Point to check
    :return True if point is inside the rectangle

    """
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def center_of_triangle(triangle):
    """
    Calculate center of triangle (p1,p2,p3)
    Triangle [p1, p2, p3]
    :return center, (x,y) in integer
    """
    x = int((triangle[0][0] + triangle[1][0] + triangle[2][0]) / 3)
    y = int((triangle[0][1] + triangle[1][1] + triangle[2][1]) / 3)
    return x, y


def distance_between_points(p1, p2):
    """
    Calculate distance between two points
    :return integer, distance
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    d = math.sqrt(dx ** 2 + dy ** 2)
    return int(d)


def calculate_line(p1, p2):
    """
    Calcalute f(x) = ax + b for the specified punts
    :param p1: point (x,y)
    :param p2: point (x,y)
    :return (a,b) specifying line ax+b that connects both points
    """
    # Calculate f = ax +b for center eyes
    (x1, y1) = p1
    (x2, y2) = p2
    a = (y2 - y1) / (x2 - x1)
    b = y1 - (a * x1)
    return a, b


def point_inside_triangle(point, triangle):
    """
    Determine if the specified point is located inside
    the given triangle (polygon)
    :param point Point (x,y) to check
    :param triangle Triangle to check [p1, p2, p3] p in form (x,y)
    :return True if the point is inside the triangle
    """
    path = mplot_path.Path(triangle)
    return path.contains_points([point])


def dist_point_line(p1, p2, p3):
    """
    Calculate the  distance between point 3 and the straight
    line between point 1 and 2. Points are specifief as (x,y)
    :param p1: First point of the line
    :param p2: Second  point of the line
    :param p3: Point for which the distance will be calculated
    :return distance between p3 and the line p1-p2
    """
    # Calculate f = ax +b for center eyes
    (x1, y1) = p1
    (x2, y2) = p2
    (x0, y0) = p3
    return abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))


def fit_ellipse(cont):
    """
    Fit ellipse to the given contour. Fitzgibbon approach
    Result:
    cx - center x
    cy - center y
    a - a
    b - b
    angle - Angle of rotation
    :param cont: countour, list of points
    :return: params describing the ellipse in the form [cx,cy,a,b,angle]
    """

    x = cont[:, 0]
    y = cont[:, 1]
    x = x[:, None]
    y = y[:, None]

    D = np.hstack([x * x, x * y, y * y, x, y, np.ones(x.shape)])
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))

    n = np.argmax(E)
    a = V[:, n]

    # Fit ellipse
    b, c, d, f, g, a = a[1]/2., a[2], a[3]/2., a[4]/2., a[5], a[0]
    num = b * b - a * c
    cx = (c * d - b * f) / num
    cy = (a*f-b*d)/num

    angle = 0.5 * np.arctan(2 * b / (a - c)) * 180 / np.pi
    up = 2 * (a * f * f + c * d * d + g * b * b-2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    a = np.sqrt(abs(up / down1))
    b = np.sqrt(abs(up / down2))

    params = [cx, cy, a, b, angle]
    return params
