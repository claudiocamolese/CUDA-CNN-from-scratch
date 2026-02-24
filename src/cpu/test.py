from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss

class Tester:
    def __init__(self, test_loader, device):
        """Initialize test loader, device and loss function

        Args:
            test_loader (pytorch.dataloader): TestLoader used during testing
            device (pytorch.device): cpu or cuda
        """
        self.test_loader = test_loader
        self.device = device
        self.criterion = CrossEntropyLoss()

    @torch.no_grad()
    def test(self, model):
        """Test method

        Args:
            model (pytorch model): model to be tested
        """
        model.to(self.device)
        model.eval()

        running_loss = 0.0
        num_batches = len(self.test_loader)

        loop = tqdm(self.test_loader, desc="Testing")

        all_preds = []
        all_labels = []

        for img, label in loop:
            img = img.to(self.device)
            label = label.to(self.device)

            pred = model(img)

            if self.criterion is not None:
                
                if isinstance(self.criterion, CrossEntropyLoss):
                    label = label.view(-1).long()

                loss = self.criterion(pred, label)
                running_loss += loss.item()
                loop.set_postfix(loss=running_loss / (loop.n + 1))

            all_preds.append(pred.cpu())
            all_labels.append(label.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        avg_loss = running_loss / num_batches if self.criterion is not None else None

        if avg_loss is not None:
            print(f"Test avg loss: {avg_loss:.6f}")

        predicted_classes = all_preds.argmax(dim=1)
        correct = (predicted_classes == all_labels).sum().item()
        total = all_labels.size(0)
        accuracy = 100 * correct / total
        print(f"Test accuracy = {accuracy:.2f}%")

