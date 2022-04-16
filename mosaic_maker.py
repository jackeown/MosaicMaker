import os
import PIL.Image
import numpy as np
from glob import glob
from rich.progress import track
import IPython
import scipy.ndimage


def loadImage(path):
    image = PIL.Image.open(path)
    image = np.array(image)
    image = image[..., :3]

    if len(image.shape) < 3 or image.shape[2] < 3:
        return None

    return image


def middle_color(image):
    s = image.shape
    return image[s[0]//2, s[1]//2]

def avg_color(image):
    return image.mean(0).mean(0)

def show(image):
    PIL.Image.fromarray(image).show()



def constructTile(image, width, height):
    tile_horizontalness = width / height
    image_horizontalness = image.shape[1] / image.shape[0]
    
    if tile_horizontalness > image_horizontalness:
        left, right = (0, image.shape[1])
        up, down = (0, int(image.shape[1] / tile_horizontalness))
    else:
        left, right = (0, int(image.shape[0] * tile_horizontalness))
        up, down = (0, image.shape[0])

    cropped = image[up:down, left:right]
    sampled = np.zeros([height, width, 3], dtype=np.uint8)
    
    y_size = cropped.shape[0] // height
    x_size = cropped.shape[1] // width
    for i in range(height):
        for j in range(width):
            sampled[i,j, :] = cropped[i*y_size, j*x_size]
    return sampled


def tileGenerator(images, tile_width, tile_height):
    while True:
        for image in images:
            yield constructTile(image, tile_width, tile_height)



def constructMosaic(src_images, target_image, tile_width, tile_height):
    src_images = [img[..., :3] for img in src_images if len(img.shape) == 3 and img.shape[2] >= 3]
    mosaic = np.zeros_like(target_image)

    tileGen = tileGenerator(src_images, tile_width, tile_height)
    # tiles = np.array([next(tileGen) for _ in track(range(20), description="Making Tiles...")])
    tiles = np.array([next(tileGen) for _ in track(range((mosaic.shape[0] // tile_height) * (mosaic.shape[1] // tile_width)), description="Making Tiles...")])
    tileColors = np.array([avg_color(tile) for tile in track(tiles, description="Getting Tile Colors...")])

    for i in track(range(0, mosaic.shape[0], tile_height), description="Placing Tiles"):
        for j in range(0, mosaic.shape[1], tile_width):
            targetColor = avg_color(target_image[i:i+tile_height, j:j+tile_width])
            index = np.argmin(np.linalg.norm(tileColors - targetColor, axis=1))
            tile = tiles[index]
            if i + tile_height < mosaic.shape[0] and j+tile_width < mosaic.shape[1]:
                mosaic[i:i+tile_height, j:j+tile_width] = tile
    
    # IPython.embed()

    return mosaic


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("src_images", help="path to folder containing images you'd like to translate into a mosaic")
    parser.add_argument("target_image", help="path to an image to use as the target")
    parser.add_argument("outfile", default="outfile.png", help="path you would like the mosaic to be written to")
    parser.add_argument("--width_of_tiles", type=int, default=40, help="width of the mosaic tiles in pixels")
    parser.add_argument("--height_of_tiles", type=int, default=40, help="height of the mosaic tiles in pixels")
    args = parser.parse_args()



    # Load in images 
    src_images = glob(os.path.join(args.src_images, "*"))
    src_images = [loadImage(path) for path in track(src_images, description="Loading Images...")]
    src_images = [x for x in src_images if x is not None]
    target_image = loadImage(args.target_image)
    s = 2500 // max(target_image.shape)
    print(f"scaling {args.target_image} by a factor of {s}...")
    target_image = scipy.ndimage.zoom(target_image, (s,s,1))

    # Construct the mosaic as a numpy ndarray
    mosaic = constructMosaic(src_images, target_image, args.width_of_tiles, args.height_of_tiles)

    # show(mosaic)

    # Write mosaic to args.outfile
    im = PIL.Image.fromarray(mosaic)
    im.save(args.outfile)

