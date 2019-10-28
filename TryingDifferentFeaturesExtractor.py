#%% Import
import cv2
import time
import matplotlib.pyplot as plt
#%% Function definition
def test_feature_detector(detector, imfname):
    image = cv2.imread(imfname)
    forb = cv2.FeatureDetector_create(detector)
    # Detect crashes program if image is not greyscale
    t1 = time.time()
    kpts,desc = forb.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    t2 = time.time()
    print(detector, 'number of KeyPoint objects', len(kpts), '(time', t2-t1, ')')

    return kpts,desc
def test_matcher(desc1, desc2):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(desc1, desc2, k= 2)
    # keep good matches:
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good.append([m])
    return good
#%%
# Parameters setting
reference_image_path = 'BaseDocument/Data/templates/ic/ita/CIC/front.png'
test_image_path = 'FronteCI.jpg'
# Opening images
reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
# Features extractor definition
print('Create SIFT')
#sift = cv2.xfeatures2d.SIFT_create()
print('Create SURF')
#surf = cv2.xfeatures2d.SURF_create()
print('Create brief')
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
print('Create freak')
freak = cv2.xfeatures2d.FREAK_create()
print('Create start')
star = cv2.xfeatures2d.StarDetector_create()
print('Create fast')
fast = cv2.FastFeatureDetector_create()
print('Create orb')
orb = cv2.ORB_create()
print('Create akaze')
akaze = cv2.AKAZE_create()
# Matcher
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
# key point extractor
print('use SIFT')
# sift_kpts1, sift_desc1 = sift.detectAndCompute(reference_image, None)
# sift_kpts2, sift_desc2 = sift.detectAndCompute(test_image, None)
# sift_matches = matcher.knnMatch(sift_desc1,sift_desc2, k= 2)
# Apply ratio test
# sift_good = []
# for m,n in sift_matches:
#     if m.distance < 0.9*n.distance:
#         sift_goodgood.append([m])
# sift_result = cv2.drawMatchesKnn(reference_image, sift_kpts1, test_image, sift_kpts2, sift_matches, None, flags=2)
# cv2.imwrite('SIFT_feature_extractor.png', sift_result)
# cv2.imshow('SIFT_features extractor', sift_result)
# cv2.waitKey(0)
print('use SURF')
# surf_kpts1, surf_desc1 = surf.detectAndCompute(reference_image, None)
# surf_kpts2, surf_desc2 = surf.detectAndCompute(test_image, None)
# surf_matches = matcher.knnMatch(surf_desc1,surf_desc2, k= 2)
# surf_good = []
# for m,n in surf_matches:
#     if m.distance < 0.9*n.distance:
#         surf_goodgood.append([m])
# surf_result = cv2.drawMatchesKnn(reference_image, surf_kpts1, test_image, surf_kpts2, surf_matches, None, flags=2)
# cv2.imwrite('surf_feature_extractor.png', surf_result)
# cv2.imshow('surf_features extractor', surf_result)
# cv2.waitKey(0)
# print('use brief')
# brief_kpts1, brief_desc1 = brief.detectAndCompute(reference_image, None)
# brief_kpts2, brief_desc2 = brief.detectAndCompute(test_image, None)
# brief_matches = matcher.knnMatch(brief_desc1,brief_desc2, k= 2)
# brief_good = []
# for m,n in brief_matches:
#     if m.distance < 0.9*n.distance:
#         brief_goodgood.append([m])
# brief_result = cv2.drawMatchesKnn(reference_image, brief_kpts1, test_image, brief_kpts2, brief_matches, None, flags=2)
# cv2.imwrite('brief_feature_extractor.png', brief_result)
# cv2.imshow('brief_features extractor', brief_result)
# cv2.waitKey(0)
# print('use freak')
# freak_kpts1, freak_desc1 = freak.detectAndCompute(reference_image, None)
# freak_kpts2, freak_desc2 = freak.detectAndCompute(test_image, None)
# freak_matches = matcher.knnMatch(freak_desc1,freak_desc2, k= 2)
# freak_good = []
# for m,n in freak_matches:
#     if m.distance < 0.9*n.distance:
#         freak_goodgood.append([m])
# freak_result = cv2.drawMatchesKnn(reference_image, freak_kpts1, test_image, freak_kpts2, freak_matches, None, flags=2)
# cv2.imwrite('freak_feature_extractor.png', freak_result)
# cv2.imshow('freak_features extractor', freak_result)
# cv2.waitKey(0)
# print('use star')
# star_kpts1, star_desc1 = star.detectAndCompute(reference_image, None)
# star_kpts2, star_desc2 = star.detectAndCompute(test_image, None)
# star_matches = matcher.knnMatch(star_desc1,star_desc2, k= 2)
# star_good = []
# for m,n in star_matches:
#     if m.distance < 0.9*n.distance:
#         star_goodgood.append([m])
# star_result = cv2.drawMatchesKnn(reference_image, star_kpts1, test_image, star_kpts2, star_matches, None, flags=2)
# cv2.imwrite('star_feature_extractor.png', star_result)
# cv2.imshow('star_features extractor', star_result)
# cv2.waitKey(0)
# print('use fast')
# fast_kpts1, fast_desc1 = fast.detectAndCompute(reference_image, None)
# fast_kpts2, fast_desc2 = fast.detectAndCompute(test_image, None)
# fast_matches = matcher.knnMatch(fast_desc1,fast_desc2, k= 2)
# fast_good = []
# for m,n in fast_matches:
#     if m.distance < 0.9*n.distance:
#         fast_goodgood.append([m])
# fast_result = cv2.drawMatchesKnn(reference_image, fast_kpts1, test_image, fast_kpts2, fast_matches, None, flags=2)
# cv2.imwrite('fast_feature_extractor.png', fast_result)
# cv2.imshow('fast_features extractor', fast_result)
# cv2.waitKey(0)
print('use orb')
orb_kpts1, orb_desc1 = orb.detectAndCompute(reference_image, None)
orb_kpts2, orb_desc2 = orb.detectAndCompute(test_image, None)
orb_matches = matcher.knnMatch(orb_desc1,orb_desc2, k= 2)
orb_good = []
for m,n in orb_matches:
    if m.distance < 0.9*n.distance:
        orb_good.append([m])
orb_result = cv2.drawMatchesKnn(reference_image, orb_kpts1, test_image, orb_kpts2, orb_good, None, flags=2)
cv2.imwrite('orb_feature_extractor.png', orb_result)
cv2.imshow('orb_features extractor', orb_result)
cv2.waitKey(0)
print('use akaze')
akaze_kpts1, akaze_desc1 = akaze.detectAndCompute(reference_image, None)
akaze_kpts2, akaze_desc2 = akaze.detectAndCompute(test_image, None)
akaze_matches = matcher.knnMatch(akaze_desc1,akaze_desc2, k= 2)
akaze_good = []
for m,n in akaze_matches:
    if m.distance < 0.9*n.distance:
        akaze_good.append([m])
akaze_result = cv2.drawMatchesKnn(reference_image, akaze_kpts1, test_image, akaze_kpts2, akaze_good, None, flags=2)
cv2.imwrite('akaze_feature_extractor.png', akaze_result)
cv2.imshow('akaze_features extractor', akaze_result)
cv2.waitKey(0)