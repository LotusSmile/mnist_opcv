'''
Author : LotusSmile
Reference : https://docs.opencv.org/3.4/dd/d3b/tutorial_py_svm_opencv.html
'''
import cv2 as cv
import numpy as np

SZ = 20 # each character(hand-written digit) size
bin_n = 16  # 360 degree / 15 = 16 bins -> (0, 1, 2, ... 15)
affine_flags = cv.WARP_INVERSE_MAP | cv.INTER_LINEAR
# rectangle -> parallelogram
# WARP_INVERSE_MAP : inverse transformation, destination -> source
# INTER_LINEAR : linear interpolation

def deskew(img):
    m = cv.moments(img) # get moments of img
    if abs(m['mu02']) < 1e-2: # abs : get absolute value
        return img.copy()
    skew = m['mu11']/m['mu02'] # get skewness
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]]) # matrix M for affine transformation
    img2 = cv.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
    # warpAffine : src -> dst using matrix M with INTER_LINEAR flag
    return img2

def hog(img2):
    gx = cv.Sobel(img2, cv.CV_32F, 1, 0) # CV_32F : 32bit float
    gy = cv.Sobel(img2, cv.CV_32F, 0, 1)
    # find Sobel derivatives of each cell in X and Y direction
    mag, ang = cv.cartToPolar(gx, gy) # each gx, gy, mag, ang is 20
    #  find their magnitude and direction of gradient at each pixel
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...15)
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    # img -> 4 parts
    hists = [np.bincount(b.ravel(), weights=m.ravel(), minlength=bin_n) for b, m in zip(bin_cells, mag_cells)]
    # bincount : count factors,  ravel : flattening
    # hists = (?, 16, 4)
    hist = np.hstack(hists)     # hist is a 64 bit vector (?, 64)
    return hist

img = cv.imread('digits.png', 0) # 0: gray, 1: color, -1: alpha channel
# (1000, 2000) -> row: 각 숫자 5개씩 50개 / column: 100 ? 개
# 숫자 1개에 row 20, column 20

if img is None:
    raise Exception("we need the digits.png image from samples/data here !")
cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]
# (50, 100, 20, 20)
# hsplit(열방향 100등분),              vsplit(행방향 50등분)

# First half is trainData, remaining is testData
train_cells = np.array([i[:50] for i in cells]) # (50, 50, 20, 20)
test_cells = np.array([i[50:] for i in cells]) # (50, 50, 20, 20)

deskewed = []
for row in train_cells:
    for col in row:
        deskewed.append(deskew(col))
deskewed = np.array(deskewed).reshape((50, 50, 20, 20))

hogdata = []
for row in deskewed:
    for col in row:
        hogdata.append(hog(col))

hogdata = np.array(hogdata, dtype=np.float32)
trainData = hogdata

responses = np.repeat(np.arange(10), 250)[:, np.newaxis]
# Real target value,  0~9의 값을 250번 반복 하되 각 행에 값이 열 하나씩

svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setType(cv.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)
svm.train(trainData, cv.ml.ROW_SAMPLE, responses)
svm.save('svm_data.dat')

deskewed_test = []
for row in train_cells:
    for col in row:
        deskewed_test.append(deskew(col))
deskewed_test = np.array(deskewed).reshape((50, 50, 20, 20))

hogdata_test = []
for row in deskewed:
    for col in row:
        hogdata_test.append(hog(col))
hogdata_test = np.array(hogdata, dtype=np.float32)
testData = hogdata

result = svm.predict(testData)[1]
# Predicted target value

mask = []
for y_hat, y in zip(result, responses):
    if y_hat == y:
        mask.append(1)
    else:
        mask.append(0)

correct = np.count_nonzero(mask)
print('Test Accuracy : {:} %'.format(correct*100.0/result.size))