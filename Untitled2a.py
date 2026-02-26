{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMHt+KmotNjWqhlRupX/C/N",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/zubeyir-donmez/Python/blob/main/Untitled2a.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ea12556d"
      },
      "source": [
        "import cv2\n",
        "import numpy as np\n",
        "\n",
        "# Create a black image (50x50 pixels)\n",
        "img = np.zeros((50, 50, 3), dtype=np.uint8)\n",
        "\n",
        "# Draw a white rectangle on the image\n",
        "cv2.rectangle(img, (10, 10), (40, 40), (255, 255, 255), -1)\n",
        "\n",
        "# Print a message to confirm cv2 is working\n",
        "print(\"cv2 imported and used successfully!\")"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}