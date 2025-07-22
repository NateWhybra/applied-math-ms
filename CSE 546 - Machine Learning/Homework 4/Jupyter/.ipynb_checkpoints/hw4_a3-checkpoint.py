{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 124,
   "id": "2a01ca12-66fe-4394-b428-28264997edd9",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from torch import nn\n",
    "import numpy as np\n",
    "from typing import Tuple, Union, List, Callable\n",
    "from torch.optim import Adam\n",
    "import torchvision\n",
    "from torch.utils.data import DataLoader, TensorDataset, random_split\n",
    "import matplotlib.pyplot as plt\n",
    "from tqdm.notebook import tqdm\n",
    "\n",
    "import os\n",
    "os.environ[\"KMP_DUPLICATE_LIB_OK\"] = \"TRUE\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "74067f05-90dc-42d1-89f3-ec795a6198e0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "cuda\n"
     ]
    }
   ],
   "source": [
    "# Connect to GPU.\n",
    "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "print(DEVICE)  # This should print out CUDA."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 172,
   "id": "cbfb334d-c46c-4c7d-b05b-42198c5d2670",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load data.\n",
    "train_dataset = torchvision.datasets.CIFAR10(\"./data\", train=True, download=True, transform=torchvision.transforms.ToTensor())\n",
    "test_dataset = torchvision.datasets.CIFAR10(\"./data\", train=False, download=True, transform=torchvision.transforms.ToTensor())\n",
    "SAMPLE_DATA = True # TODO: Change this back!\n",
    "batch_size = 128\n",
    "# batch_size = 256\n",
    "\n",
    "if SAMPLE_DATA:\n",
    "  train_dataset, _ = random_split(train_dataset, [int(0.1 * len(train_dataset)), int(0.9 * len(train_dataset))]) # get 10% of train dataset and \"throw away\" the other 90%\n",
    "\n",
    "train_dataset, val_dataset = random_split(train_dataset, [int(0.9 * len(train_dataset)), int( 0.1 * len(train_dataset))])\n",
    "\n",
    "# Create separate dataloaders for the train, test, and validation set\n",
    "train_loader = DataLoader(\n",
    "    train_dataset,\n",
    "    batch_size=batch_size,\n",
    "    shuffle=True\n",
    ")\n",
    "\n",
    "val_loader = DataLoader(\n",
    "    val_dataset,\n",
    "    batch_size=batch_size,\n",
    "    shuffle=True\n",
    ")\n",
    "\n",
    "test_loader = DataLoader(\n",
    "    test_dataset,\n",
    "    batch_size=batch_size,\n",
    "    shuffle=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "id": "066f69a8-ef15-4f20-80ad-5dd8ed164998",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Look at data.\n",
    "data_iter = iter(train_loader)\n",
    "images, labels = next(data_iter)\n",
    "image = images[0]\n",
    "image = image.permute(1, 2, 0)\n",
    "plt.imshow(image)\n",
    "plt.show()\n",
    "\n",
    "# Get d.\n",
    "d = image.shape.numel()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 176,
   "id": "5db5488d-bb00-4659-aff0-6aeaf495c09c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Definie the first model.\n",
    "def model_A(M=1500) -> nn.Module:\n",
    "    def model_func():\n",
    "        \"\"\"Instantiate a linear model and send it to device.\"\"\"\n",
    "        model = nn.Sequential(\n",
    "                nn.Flatten(),\n",
    "                nn.Linear(in_features=d, out_features=M),\n",
    "                nn.ReLU(),\n",
    "                nn.Linear(M, 10)\n",
    "             )\n",
    "        return model.to(DEVICE)\n",
    "    return model_func"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 188,
   "id": "36795a29-db3f-429a-8a68-331ea0228d08",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ...\n",
    "def train_A(\n",
    "    model: nn.Module, optimizer: SGD,\n",
    "    train_loader: DataLoader, val_loader: DataLoader,\n",
    "    epochs: int = 20\n",
    "    ) -> Tuple[List[float], List[float], List[float], List[float]]:\n",
    "    \"\"\"\n",
    "    Trains a model for the specified number of epochs using the loaders.\n",
    "\n",
    "    Returns:\n",
    "    Lists of training loss, training accuracy, validation loss, validation accuracy for each epoch.\n",
    "    \"\"\"\n",
    "\n",
    "    loss = nn.CrossEntropyLoss()\n",
    "    train_losses = []\n",
    "    train_accuracies = []\n",
    "    val_losses = []\n",
    "    val_accuracies = []\n",
    "    for e in tqdm(range(epochs)):\n",
    "        model.train()\n",
    "        train_loss = 0.0\n",
    "        train_acc = 0.0\n",
    "\n",
    "        # Main training loop; iterate over train_loader. The loop\n",
    "        # terminates when the train loader finishes iterating, which is one epoch.\n",
    "        for (x_batch, labels) in train_loader:\n",
    "            x_batch, labels = x_batch.to(DEVICE), labels.to(DEVICE)\n",
    "            optimizer.zero_grad()\n",
    "            labels_pred = model(x_batch)\n",
    "            batch_loss = loss(labels_pred, labels)\n",
    "            train_loss = train_loss + batch_loss.item()\n",
    "\n",
    "            labels_pred_max = torch.argmax(labels_pred, 1)\n",
    "            batch_acc = torch.sum(labels_pred_max == labels)\n",
    "            train_acc = train_acc + batch_acc.item()\n",
    "\n",
    "            batch_loss.backward()\n",
    "            optimizer.step()\n",
    "        train_losses.append(train_loss / len(train_loader))\n",
    "        train_accuracies.append(train_acc / (batch_size * len(train_loader)))\n",
    "\n",
    "        # Validation loop; use .no_grad() context manager to save memory.\n",
    "        model.eval()\n",
    "        val_loss = 0.0\n",
    "        val_acc = 0.0\n",
    "\n",
    "        with torch.no_grad():\n",
    "            for (v_batch, labels) in val_loader:\n",
    "                v_batch, labels = v_batch.to(DEVICE), labels.to(DEVICE)\n",
    "                labels_pred = model(v_batch)\n",
    "                v_batch_loss = loss(labels_pred, labels)\n",
    "                val_loss = val_loss + v_batch_loss.item()\n",
    "\n",
    "                v_pred_max = torch.argmax(labels_pred, 1)\n",
    "                batch_acc = torch.sum(v_pred_max == labels)\n",
    "                val_acc = val_acc + batch_acc.item()\n",
    "            val_losses.append(val_loss / len(val_loader))\n",
    "            val_accuracies.append(val_acc / (batch_size * len(val_loader)))\n",
    "            # print(\"Val Acc\", val_acc / (batch_size * len(val_loader)))\n",
    "\n",
    "    return train_losses, train_accuracies, val_losses, val_accuracies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 204,
   "id": "8d2b9b80-c212-4676-9920-350f529e273b",
   "metadata": {},
   "outputs": [],
   "source": [
    "def parameter_search(train_loader: DataLoader,\n",
    "                     val_loader: DataLoader,\n",
    "                     model_fn) -> float:\n",
    "    \"\"\"\n",
    "    Parameter search for our linear model using SGD.\n",
    "\n",
    "    Args:\n",
    "    train_loader: the train dataloader.\n",
    "    val_loader: the validation dataloader.\n",
    "    model_fn: a function that, when called, returns a torch.nn.Module.\n",
    "\n",
    "    Returns:\n",
    "    The learning rate with the least validation loss.\n",
    "    NOTE: You may need to modify this function to search over and return\n",
    "     other parameters beyond learning rate.\n",
    "    \"\"\"\n",
    "    best_loss = torch.tensor(np.inf)\n",
    "    best_lr = 0.0\n",
    "    best_decay = 0\n",
    "    best_momentum = 0\n",
    "    best_acc = 0\n",
    "\n",
    "    lrs = np.random.uniform(1e-3, 1e-2, 5)\n",
    "    decays = np.random.uniform(1e-5, 1e-3, 5)\n",
    "    momentums = [0.6, 0.75, 0.9]\n",
    "     \n",
    "    for lr in lrs:\n",
    "        for decay in decays:\n",
    "            for momentum in momentums:\n",
    "                print(f\"trying learning rate {lr}\")\n",
    "                print(f\"trying L2 reg {decay}\")\n",
    "                print(f\"trying momentum {momentum}\")\n",
    "                # print(f\"Hidden {h}\")\n",
    "                model = model_fn()\n",
    "                optim = SGD(model.parameters(), lr, momentum=momentum, weight_decay=decay)\n",
    "                train_loss, train_acc, val_loss, val_acc = train_A(\n",
    "                    model,\n",
    "                    optim,\n",
    "                    train_loader,\n",
    "                    val_loader,\n",
    "                    epochs=15\n",
    "                    )\n",
    "\n",
    "                print(max(val_acc))\n",
    "        \n",
    "                if min(val_loss) < best_loss:\n",
    "                    best_loss = min(val_loss)\n",
    "                    best_lr = lr\n",
    "                    best_decay = decay\n",
    "                    best_momentum = momentum\n",
    "                    best_acc = max(val_acc)\n",
    "\n",
    "                if max(val_acc) >= 0.5:\n",
    "                    best_loss = min(val_loss)\n",
    "                    best_lr = lr\n",
    "                    best_decay = decay\n",
    "                    best_momentum = momentum\n",
    "                    best_acc = max(val_acc)\n",
    "                    break\n",
    "                    \n",
    "    return (best_lr, best_decay, best_momentum, best_acc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 205,
   "id": "6fa7f0fa-8bc8-42bc-bf76-8ccce6e7a03e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "trying learning rate 0.009562343597344876\n",
      "trying L2 reg 0.00048167144051312655\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "28b5e734849d4213853612af98db2b43",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.357421875\n",
      "trying learning rate 0.009562343597344876\n",
      "trying L2 reg 0.00048167144051312655\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1b60d146074942bb82165bc0c221c957",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.400390625\n",
      "trying learning rate 0.009562343597344876\n",
      "trying L2 reg 0.00048167144051312655\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5d9ccbad665a4fa3847cc67a8c202fc9",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.396484375\n",
      "trying learning rate 0.009562343597344876\n",
      "trying L2 reg 0.0006460245405125394\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "67feb093a2344a559f9b4d3dd2b3b06d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.390625\n",
      "trying learning rate 0.009562343597344876\n",
      "trying L2 reg 0.0006460245405125394\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "28e5e38de8d74c00b2dbbd19611a134e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.388671875\n",
      "trying learning rate 0.009562343597344876\n",
      "trying L2 reg 0.0006460245405125394\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6ef05b116c35439398be7ece7f53e145",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.404296875\n",
      "trying learning rate 0.009562343597344876\n",
      "trying L2 reg 0.0007221286759314962\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "684b960eeb5f40d58bf4ae4818d559cd",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.3828125\n",
      "trying learning rate 0.009562343597344876\n",
      "trying L2 reg 0.0007221286759314962\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "50fe3843a06f4ed2a39436d99b752259",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.380859375\n",
      "trying learning rate 0.009562343597344876\n",
      "trying L2 reg 0.0007221286759314962\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "894d66d5ec5243caae023d8ab21e5b42",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.4140625\n",
      "trying learning rate 0.009562343597344876\n",
      "trying L2 reg 0.0009848902832969115\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9aa3af959eac413ab81d41be7817f840",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.376953125\n",
      "trying learning rate 0.009562343597344876\n",
      "trying L2 reg 0.0009848902832969115\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d5997e8c78c44d3cbc1f15084a77e216",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.38671875\n",
      "trying learning rate 0.009562343597344876\n",
      "trying L2 reg 0.0009848902832969115\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "895616ac3c5543f7a0d489ec81df5c25",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.408203125\n",
      "trying learning rate 0.009562343597344876\n",
      "trying L2 reg 0.0004856692868949005\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a71858f0b26f4e4ab7db84966dd5ae30",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.380859375\n",
      "trying learning rate 0.009562343597344876\n",
      "trying L2 reg 0.0004856692868949005\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "01adb0d55d254352aebad93929604a58",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.41015625\n",
      "trying learning rate 0.009562343597344876\n",
      "trying L2 reg 0.0004856692868949005\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "442fa75c55c4402a83eefbadcf36a025",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.4140625\n",
      "trying learning rate 0.005317796185043715\n",
      "trying L2 reg 0.00048167144051312655\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "941ae097443a4e4fa16a0a3d4c2a33ba",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.36328125\n",
      "trying learning rate 0.005317796185043715\n",
      "trying L2 reg 0.00048167144051312655\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6d06bb0fcb464bd093120051ff3b780d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.38671875\n",
      "trying learning rate 0.005317796185043715\n",
      "trying L2 reg 0.00048167144051312655\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b305ff28af92462883294e4c226f72ea",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.396484375\n",
      "trying learning rate 0.005317796185043715\n",
      "trying L2 reg 0.0006460245405125394\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e473de54e64147a8b64e3f0f9385a589",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.380859375\n",
      "trying learning rate 0.005317796185043715\n",
      "trying L2 reg 0.0006460245405125394\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "67ce7b7c149443daa5b3671c9079171a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.3828125\n",
      "trying learning rate 0.005317796185043715\n",
      "trying L2 reg 0.0006460245405125394\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3d74fc09b14640a99a4f72aa6cf13ba3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.4140625\n",
      "trying learning rate 0.005317796185043715\n",
      "trying L2 reg 0.0007221286759314962\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d7a53edbe53c42c29466edd20f0ab994",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.359375\n",
      "trying learning rate 0.005317796185043715\n",
      "trying L2 reg 0.0007221286759314962\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "07ebac451b2c4494b48ed17766b13bbb",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.388671875\n",
      "trying learning rate 0.005317796185043715\n",
      "trying L2 reg 0.0007221286759314962\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "35fd6ddcf4e249008f4b2abbd3364a8d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.419921875\n",
      "trying learning rate 0.005317796185043715\n",
      "trying L2 reg 0.0009848902832969115\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8a96616abb824a38aa52973dd9a26699",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.37890625\n",
      "trying learning rate 0.005317796185043715\n",
      "trying L2 reg 0.0009848902832969115\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2e1f17635a2e4c33bc56dd148343730b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.3828125\n",
      "trying learning rate 0.005317796185043715\n",
      "trying L2 reg 0.0009848902832969115\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e7e121e865fc4488b9db9270bc9b0d41",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.41015625\n",
      "trying learning rate 0.005317796185043715\n",
      "trying L2 reg 0.0004856692868949005\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6b4c390eb8614f3bb3e99fa78d21b8af",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.369140625\n",
      "trying learning rate 0.005317796185043715\n",
      "trying L2 reg 0.0004856692868949005\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "645b95d6e2444c19832ea30d3e938de1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.412109375\n",
      "trying learning rate 0.005317796185043715\n",
      "trying L2 reg 0.0004856692868949005\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "36dcf467d52841448eae4ea8d7ba4d31",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.40625\n",
      "trying learning rate 0.004505927646052294\n",
      "trying L2 reg 0.00048167144051312655\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ee97670197494878ab8f6177042cb1f2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.369140625\n",
      "trying learning rate 0.004505927646052294\n",
      "trying L2 reg 0.00048167144051312655\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6376494e0f614984b15666d1bc42cf5a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.39453125\n",
      "trying learning rate 0.004505927646052294\n",
      "trying L2 reg 0.00048167144051312655\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a303e7395d664b6d8fd2354181bace87",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.41015625\n",
      "trying learning rate 0.004505927646052294\n",
      "trying L2 reg 0.0006460245405125394\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "351c6191c86645df851d2834f10c1921",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.380859375\n",
      "trying learning rate 0.004505927646052294\n",
      "trying L2 reg 0.0006460245405125394\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8e6cda30211d464298585ef91216579f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.3828125\n",
      "trying learning rate 0.004505927646052294\n",
      "trying L2 reg 0.0006460245405125394\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "671696a865e5416eacbd2d9cae3e7ad7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.41015625\n",
      "trying learning rate 0.004505927646052294\n",
      "trying L2 reg 0.0007221286759314962\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "cd127c9754c143abbe919b6d1384b1ed",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.36328125\n",
      "trying learning rate 0.004505927646052294\n",
      "trying L2 reg 0.0007221286759314962\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3bca3fd464364db295a259b0ca97cbef",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.3828125\n",
      "trying learning rate 0.004505927646052294\n",
      "trying L2 reg 0.0007221286759314962\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b4004808354248f9904a30a63e8fd949",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.408203125\n",
      "trying learning rate 0.004505927646052294\n",
      "trying L2 reg 0.0009848902832969115\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d2868d379aa7462b80a94706128abf6f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.3671875\n",
      "trying learning rate 0.004505927646052294\n",
      "trying L2 reg 0.0009848902832969115\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "70efdda84c9b47d5af54db38dfa772c2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.38671875\n",
      "trying learning rate 0.004505927646052294\n",
      "trying L2 reg 0.0009848902832969115\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3ea9d9af6eea441299c58fccbd7ffb0d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.412109375\n",
      "trying learning rate 0.004505927646052294\n",
      "trying L2 reg 0.0004856692868949005\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "43a8cc76fccb47bd8f93d06066eb098c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.3828125\n",
      "trying learning rate 0.004505927646052294\n",
      "trying L2 reg 0.0004856692868949005\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1638863fcecf4debb18f4d463f85bcb1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.404296875\n",
      "trying learning rate 0.004505927646052294\n",
      "trying L2 reg 0.0004856692868949005\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f63e80f5d1c641f588476b1946fa87a8",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.421875\n",
      "trying learning rate 0.0057684264810474156\n",
      "trying L2 reg 0.00048167144051312655\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b12aa0b81ce248beb845fb1c2c0f5d4b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.361328125\n",
      "trying learning rate 0.0057684264810474156\n",
      "trying L2 reg 0.00048167144051312655\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "29279d8963ac4c6bafb9b9422bb9790c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.40234375\n",
      "trying learning rate 0.0057684264810474156\n",
      "trying L2 reg 0.00048167144051312655\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a881c6d0e74143b88be8497adf65c590",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.404296875\n",
      "trying learning rate 0.0057684264810474156\n",
      "trying L2 reg 0.0006460245405125394\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "177a6d9c5a4d486cb34f74ecb9370b5b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.384765625\n",
      "trying learning rate 0.0057684264810474156\n",
      "trying L2 reg 0.0006460245405125394\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "40a77f63c3f84c53a71b23661180561f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.388671875\n",
      "trying learning rate 0.0057684264810474156\n",
      "trying L2 reg 0.0006460245405125394\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6731279472214cc9a69cec248188c986",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.404296875\n",
      "trying learning rate 0.0057684264810474156\n",
      "trying L2 reg 0.0007221286759314962\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b4d5f18cb58f45dfb2112270f640be5b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.3671875\n",
      "trying learning rate 0.0057684264810474156\n",
      "trying L2 reg 0.0007221286759314962\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "08cbde97bde94f1db296f32d510541d6",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.373046875\n",
      "trying learning rate 0.0057684264810474156\n",
      "trying L2 reg 0.0007221286759314962\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f75db81c0dbc48e085af1d7b962c46c6",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.416015625\n",
      "trying learning rate 0.0057684264810474156\n",
      "trying L2 reg 0.0009848902832969115\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "85c5c3ed08db4b279831120b7bcb9911",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.3828125\n",
      "trying learning rate 0.0057684264810474156\n",
      "trying L2 reg 0.0009848902832969115\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "87b9238c58ac4409b0850f346f611d34",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.3984375\n",
      "trying learning rate 0.0057684264810474156\n",
      "trying L2 reg 0.0009848902832969115\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b8d01f31637d4b8db6dabed0792e611b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.4140625\n",
      "trying learning rate 0.0057684264810474156\n",
      "trying L2 reg 0.0004856692868949005\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3fd58198fee346ffb6a4045bbcd8d561",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.361328125\n",
      "trying learning rate 0.0057684264810474156\n",
      "trying L2 reg 0.0004856692868949005\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b804db1c4bae4479899f990c656ac005",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.384765625\n",
      "trying learning rate 0.0057684264810474156\n",
      "trying L2 reg 0.0004856692868949005\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6e31b3c1ec49468a8a6932a123b4b229",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.416015625\n",
      "trying learning rate 0.008181856561245083\n",
      "trying L2 reg 0.00048167144051312655\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "44b036ae62ba4bb0b6a2517d002359b9",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.37890625\n",
      "trying learning rate 0.008181856561245083\n",
      "trying L2 reg 0.00048167144051312655\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "bfe67bfa85d04fb9b57cf79733add4de",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.396484375\n",
      "trying learning rate 0.008181856561245083\n",
      "trying L2 reg 0.00048167144051312655\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8855056cb1aa4992bfbcea94772180d2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.419921875\n",
      "trying learning rate 0.008181856561245083\n",
      "trying L2 reg 0.0006460245405125394\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d35a05cbf20540d0b69158efcc87648a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.384765625\n",
      "trying learning rate 0.008181856561245083\n",
      "trying L2 reg 0.0006460245405125394\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "15f2251b37f648d2a703229df37c8811",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.400390625\n",
      "trying learning rate 0.008181856561245083\n",
      "trying L2 reg 0.0006460245405125394\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "247b1c7d5efd414a82bfc09b8e83d7aa",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.40625\n",
      "trying learning rate 0.008181856561245083\n",
      "trying L2 reg 0.0007221286759314962\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9f2a3e6a46ee47d696c47beb73ace3ef",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.376953125\n",
      "trying learning rate 0.008181856561245083\n",
      "trying L2 reg 0.0007221286759314962\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6321165e948f47019ab62d1bf496a692",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.3828125\n",
      "trying learning rate 0.008181856561245083\n",
      "trying L2 reg 0.0007221286759314962\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b8bc7ee7bcac4dbb8029a989af3d3360",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.412109375\n",
      "trying learning rate 0.008181856561245083\n",
      "trying L2 reg 0.0009848902832969115\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f0787e94e41d478d8b679f7455807a68",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.388671875\n",
      "trying learning rate 0.008181856561245083\n",
      "trying L2 reg 0.0009848902832969115\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3360d62a1ac5490aabff924119355644",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.404296875\n",
      "trying learning rate 0.008181856561245083\n",
      "trying L2 reg 0.0009848902832969115\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "db0f0664a12e4d2e94bc268e2dedf986",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.3984375\n",
      "trying learning rate 0.008181856561245083\n",
      "trying L2 reg 0.0004856692868949005\n",
      "trying momentum 0.6\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8ba844d5c0864c26bcfbdc36b7f8b186",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.384765625\n",
      "trying learning rate 0.008181856561245083\n",
      "trying L2 reg 0.0004856692868949005\n",
      "trying momentum 0.75\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "560279a7823b4ca2a77c18f1a901d3d0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.392578125\n",
      "trying learning rate 0.008181856561245083\n",
      "trying L2 reg 0.0004856692868949005\n",
      "trying momentum 0.9\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a9f9e9f91031451c8f1a647be8ea2cef",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/15 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.4140625\n"
     ]
    }
   ],
   "source": [
    "best_params = parameter_search(train_loader, val_loader, model_A(M=2048))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 206,
   "id": "cefa2ce7-0e95-443b-8959-147d8780dfbe",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(0.009562343597344876, 0.0007221286759314962, 0.9, 0.4140625)\n"
     ]
    }
   ],
   "source": [
    "print(best_params)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b480c340-956e-43e6-9c89-6282f93826dd",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "cse446",
   "language": "python",
   "name": "cse446"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.21"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
