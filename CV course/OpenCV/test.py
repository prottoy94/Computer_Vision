import cv2
img=cv2.imread('test_image.jpg')

resize=cv2.resize(img,(700,500))
cv2.imshow('Resized Image',resize)

greyscale=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Greyscale Image',greyscale)

blur=cv2.GaussianBlur(img,(15,15),0)
cv2.imshow('Blurred Image',blur)

edge=cv2.Canny(img,50,150)
cv2.imshow('Edge Detection',edge)
cv2.waitKey(0)
cv2.destroyAllWindows()