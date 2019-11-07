#%% Import section
import numpy as np
import cv2
from matplotlib import pyplot as plt
#%%
references_path = 'TestImages/rear/'
reference_image = cv2.imread(references_path + 'a-pantaloni-rearCIC.jpg', cv2.IMREAD_GRAYSCALE)

name_piece = reference_image[175:225, 70:120]
statura_piece = reference_image[695:745, 70:120] # non tutta la scritta
firma_piece = reference_image[595:645, 762:812] # mantengo le stesse proporzioni della finestra
#imp_piece = reference_image[695:745, 762:812] # TODO eliminare questa riga
#%%
plt.figure()
plt.imshow(name_piece)
plt.show()
plt.figure()
plt.imshow(statura_piece)
plt.show()
plt.figure()
plt.imshow(firma_piece)
plt.show()
# plt.figure()
# plt.imshow(imp_piece)
# plt.show()
#%% Saving reference
cv2.imwrite('ReferencePiece/rear/name_piece.jpg', name_piece)
cv2.imwrite('ReferencePiece/rear/statura_piece.jpg', statura_piece)
cv2.imwrite('ReferencePiece/rear/firma_piece.jpg', firma_piece)