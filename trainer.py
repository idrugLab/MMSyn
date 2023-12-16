import torch

def training_classing(model, data1, data2, optimizer, criterion, device):
    model.train()
    total_loss = []
    for _, data in enumerate(zip(data1, data2)):
        data1 = data[0]
        data2 = data[1]
        data1 = data1.to(device)
        data2 = data2.to(device)
        y = data[0].y.view(-1, 1).to(device) # (b, 1)
        optimizer.zero_grad()
        output, _ = model(data1, data2)
        loss = criterion(output, y)
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    return (sum(total_loss) / len(total_loss))

def evaluate_test_scores(model, data1, data2, criterion, device):
    model.eval()
    total_loss = []
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    with torch.no_grad():
        for data in zip(data1, data2):
            data1 = data[0]
            data2 = data[1]
            data1 = data1.to(device)
            data2 = data2.to(device)
            y = data[0].y.view(-1, 1).to(device)
            yp1, _ = model(data1, data2)
            yp2, _ = model(data1, data2)
            y_pred = (yp1 + yp2) /2
            total_loss.append((criterion(y_pred,  y)).item())
            y_pred = torch.sigmoid(y_pred)
            y_true = y.squeeze()
            targets, pred_scores =  y_true, y_pred
            total_preds = torch.cat([total_preds, pred_scores.cpu()], dim=0)
            total_labels = torch.cat([total_labels, targets.cpu()], dim=0)
    return (sum(total_loss)/len(total_loss)), total_labels.numpy().flatten(), total_preds.numpy().flatten()
