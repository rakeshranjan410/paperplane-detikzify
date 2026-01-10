from PIL import Image, ImageDraw

def create_image():
    img = Image.new('RGB', (420, 420), color='white')
    d = ImageDraw.Draw(img)
    d.rectangle([100, 100, 300, 300], outline='black', width=5)
    d.text((150, 200), "Dummy Diagram", fill='black')
    img.save('dummy.jpg')

if __name__ == "__main__":
    create_image()
