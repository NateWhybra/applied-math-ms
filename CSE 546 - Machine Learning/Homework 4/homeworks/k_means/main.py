if __name__ == "__main__":
    from k_means import lloyd_algorithm  # type: ignore
else:
    from .k_means import lloyd_algorithm

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem

import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@problem.tag("hw4-A", start_line=1)
def main():
    """Main function of k-means problem

    Run Lloyd's Algorithm for k=10, and report 10 centers returned.

    NOTE: This code might take a while to run. For debugging purposes you might want to change:
        x_train to x_train[:10000]. Make sure to change it back before submission!
    """
    # raise NotImplementedError("Your Code Goes Here")
    (x_train, _), _ = load_dataset("mnist")
    centers, errors = lloyd_algorithm(data=x_train, num_centers=10)

    fig, ax = plt.subplots(2, 5)
    for i in range(centers.shape[0]):
        image = centers[i].reshape((28, 28))
        if i < 5:
            ax[0, i].imshow(image)
            ax[0, i].axis('off')
        else:
            ax[1, i - 5].imshow(image)
            ax[1, i - 5].axis('off')

    plt.show()

if __name__ == "__main__":
    main()
