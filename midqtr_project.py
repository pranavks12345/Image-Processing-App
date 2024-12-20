"""
DSC 20 Mid-Quarter Project
Name(s): Pranav Kumarsubha
PID(s):  17123010
"""

import numpy as np
from PIL import Image

NUM_CHANNELS = 3


# --------------------------------------------------------------------------- #

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    :return: RGBImage of given file
    :param path: filepath of image
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Save the given RGBImage instance to the given path
    :param path: filepath of image
    :param image: RGBImage object to save
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)


# --------------------------------------------------------------------------- #

# Part 1: RGB Image #
class RGBImage:
    """
    TODO: add description
    """

    def __init__(self, pixels):
        """
        TODO: add description

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        """
        # YOUR CODE GOES HERE #
        # Raise exceptions here
        self.pixels = pixels
        empty_list=[]
        for i in self.pixels:
            empty_list.append(1)
            empty_list.append(len(i))
        self.num_rows = sum(empty_list)-empty_list[-1]
        self.num_cols = empty_list[-1]
        
        if type(self.pixels)!=list:
            raise TypeError
        if len(self.pixels)==0:
            raise TypeError
        for i in range(len(self.pixels)):
            if type(self.pixels[i])!=list:
                raise TypeError
            if len(self.pixels[i])==0:
                raise TypeError
            if len(self.pixels[i])!=len(self.pixels[i-1]):
                raise TypeError
            
        for x in self.pixels:
            for a in x:
                if type(a)!=list:
                    raise TypeError
                if len(a)!=3:
                    raise TypeError
                for letter in a:
                    if letter>255 or letter<0:
                        raise ValueError
        


    def size(self):
        """
        TODO: add description

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        return (self.num_rows,self.num_cols)
    def get_pixels(self):
        """
        TODO: add description

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        
        return [[[a for a in i] for i in x] for x in self.pixels]

    def copy(self):
        """
        TODO: add description

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        return RGBImage(self.get_pixels())

    def get_pixel(self, row, col):
        """
        TODO: add description

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """
        if type(row)!=int or type(col)!=int:
            raise TypeError
        if row<0 or row>255 or col<0 or col>255:
            raise ValueError
        try:
            return tuple(self.pixels[row][col])
        except IndexError:
            raise ValueError

    def set_pixel(self, row, col, new_color):
        """
        TODO: add description

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Check that the R/G/B value with negative is unchanged
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """
        if type(row)!=int or type(col)!=int:
            raise TypeError
        if row<0 or row>255 or col<0 or col>255:
            raise ValueError
        if type(new_color)!=tuple:
            raise TypeError
        if len(new_color)!=3:
            raise TypeError
        for i in new_color:
            if type(i)!=int:
                raise TypeError
        if new_color[0]>255:
            raise ValueError
        if new_color[0] >= 0:
            self.pixels[row][col][0]=new_color[0]
        if new_color[1] >= 0:
            self.pixels[row][col][1]=new_color[1]
        if new_color[2] >= 0:
            self.pixels[row][col][2]=new_color[2]





# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    TODO: add description
    """

    def __init__(self):
        """
        TODO: add description

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0
    def get_cost(self):
        """
        TODO: add description

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        return self.cost

    def negate(self, image):
        """
        TODO: add description

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img_input = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img_input)
        >>> id(img_input) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output,
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img_input = img_read_helper('img/gradient_16x16.png')           # 2
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_negate.png')  # 3
        >>> img_negate = img_proc.negate(img_input)                         # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/gradient_16x16_negate.png', img_negate)# 6
        """
        negated_image=image.copy()
        negated_image.pixels=[[[(255-val) for val in image.pixels[row][column]]\
         for column in range(len(negated_image.pixels[row]))]\
          for row in range(len(negated_image.get_pixels()))]
        return negated_image

       
        


    def grayscale(self, image):
        """
        TODO: add description

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_gray.png')
        >>> img_gray = img_proc.grayscale(img_input)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/gradient_16x16_gray.png', img_gray)
        """
        grayscale_image=image.copy()
        grayscale_image.pixels=[[[int((sum(image.pixels[row][column]))/3)]*3\
         for column in range(len(image.pixels[row]))]\
          for row in range(len(image.get_pixels()))]

        return grayscale_image
        

    def rotate_180(self, image):
        """
        TODO: add description

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img_input)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/gradient_16x16_rotate.png', img_rotate)
        """
        rotation_image=image.copy()

        rotation_image.pixels=[i[::-1] for i in image.get_pixels()][::-1]
        return rotation_image



# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    TODO: add description
    """

    def __init__(self):
        """
        TODO: add description

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0
        self.coupons=0

    def negate(self, image):
        """
        TODO: add description

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_16x16.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_negate.png')
        >>> img_negate = img_proc.negate(img_input)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        if self.coupons>0:
            self.coupons=self.coupons-1
            super().negate(image)
        else:
            self.cost=self.cost+5
            return super().negate(image)

    def grayscale(self, image):
        """
        TODO: add description

        """
        if self.coupons>0:
            self.coupons=self.coupons-1
            return super().grayscale(image)
        else:
            self.cost=self.cost+6
            return super().grayscale(image)

    def rotate_180(self, image):
        """
        TODO: add description

        # Check that the cost is 0 after two rotation calls
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img = img_proc.rotate_180(img_input)
        >>> img_proc.get_cost()
        10
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """
        if self.coupons>0:
            self.coupons=self.coupons-1
            return super().rotate_180(image)
        else:
            if self.cost ==10:
                self.cost =0
            else:
                self.cost=self.cost+10
            return super().rotate_180(image)

    def redeem_coupon(self, amount):
        """
        TODO: add description

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img_input)
        >>> img_proc.get_cost()
        0
        """
        if amount<=0:
            raise ValueError
        if type(amount)!=int:
            raise TypeError
        self.coupons=self.coupons+amount
        if self.coupons>1:
            self.cost=self.cost
    

# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    TODO: add description
    """

    def __init__(self):
        """
        TODO: add description

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        # YOUR CODE GOES HERE #
        self.cost = 50

    def chroma_key(self, chroma_image, background_image, color):
        """
        TODO: add description

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_in = img_read_helper('img/square_16x16.png')
        >>> img_in_back = img_read_helper('img/gradient_16x16.png')
        >>> color = (255, 255, 255)
        >>> img_exp = img_read_helper('img/exp/square_16x16_chroma.png')
        >>> img_chroma = img_proc.chroma_key(img_in, img_in_back, color)
        >>> img_chroma.pixels == img_exp.pixels # Check chroma_key output
        True
        >>> img_save_helper('img/out/square_16x16_chroma.png', img_chroma)
        """
        
        if type(chroma_image)!=RGBImage:
            raise TypeError
        if type(background_image)!=RGBImage:
            raise TypeError
        if chroma_image.size()!=background_image.size():
            raise ValueError
        chro_img=chroma_image.copy()
        for row in range(len(chroma_image.get_pixels())):
            for column in range(len(chroma_image.pixels[row])):
                if chroma_image.pixels[row][column]==list(color):
                    chro_img.set_pixel(row, column,\
                     tuple(background_image.get_pixel(row, column)))
        return chro_img

    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        TODO: add description

        # Test with out-of-bounds image and position size
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/gradient_16x16.png')
        >>> x, y = (15, 0)
        >>> img_proc.sticker(img_sticker, img_back, x, y)
        Traceback (most recent call last):
        ...
        ValueError

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/gradient_16x16.png')
        >>> x, y = (3, 3)
        >>> img_exp = img_read_helper('img/exp/square_16x16_sticker.png')
        >>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
        >>> img_combined.pixels == img_exp.pixels # Check sticker output
        True
        >>> img_save_helper('img/out/square_16x16_sticker.png', img_combined)
        """
        if type(sticker_image)!=RGBImage or type(background_image)!= RGBImage:
            raise TypeError()
        if type(x_pos)!=int or type(y_pos)!=int:
            raise TypeError()
        if x_pos+sticker_image.size()[0]>background_image.size()[0]:
            raise ValueError()
        if y_pos+sticker_image.size()[0]>background_image.size()[0]:
            raise ValueError()
        if x_pos+sticker_image.size()[1]>background_image.size()[1]:
            raise ValueError()
        if y_pos+sticker_image.size()[1]>background_image.size()[1]:
            raise ValueError()
        sticky_image=background_image.copy()
        for row in range(len(sticker_image.get_pixels())):
            for column in range(len(sticker_image.pixels[row])):
                sticky_image.pixels[row+y_pos]\
                [column+x_pos]=sticker_image.pixels[row][column]
        return sticky_image


# Part 5: Image KNN Classifier #
class ImageKNNClassifier:
    """
    TODO: add description
    """

    def __init__(self, n_neighbors):
        """
        TODO: add description
        """
        self.n_neighbors=n_neighbors
        self.data=[]

    def fit(self, data):
        """
        TODO: add description
        """
        if len(self.data)>0:
            raise ValueError()
        self.data.append(data)
        if len(self.data)<=self.n_neighbors:
            raise ValueError()


    @staticmethod
    def distance(image1, image2):
        """
        TODO: add description
        """
        if type(image1)!=RGBImage or type(image2)!= RGBImage:
            raise TypeError()
        first_image=[num for x in image1.get_pixels()\
         for column in x for num in column]
        second_image=[num for x in image2.get_pixels()\
         for column in x for num in column]
        if len(first_image)!=len(second_image):
            raise ValueError
        return sum([(a1-b1)**2 for a1,b1 in zip(first_image,second_image)])**0.5

        

    @staticmethod
    def vote(candidates):
        """
        TODO: add description

        """
        candidates_dict={}
        for i in candidates:
            if i in candidates_dict:
                candidates_dict[i]=candidates_dict[i]+1
            elif i not in candidates_dict:
                candidates_dict[i]=1

    def predict(self, image):
        """
        TODO: add description
        """
        if len(self.data)==0:
            raise ValueError
        list1= [self.distance(image, x[0]) for x in self.data]

        zipped_dict = dict(zip(self.data, list1))

        sort1 = sorted(zipped_dict, key=zipped_dict.get)[:self.n_neighbors]

        return self.vote(sort1)
