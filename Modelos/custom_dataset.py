from torch.utils.data import Dataset

class CustomDataset1(Dataset):
    def __init__(self, coherence_matrices, class_labels, ages, transform=None):
        self.coherence_matrices = coherence_matrices
        self.class_labels = class_labels
        self.ages = ages
        self.transform = transform

    def __len__(self):
        return len(self.coherence_matrices)

    def __getitem__(self, idx):
        coherence_matrix = self.coherence_matrices[idx]
        class_label = self.class_labels[idx]
        age = self.ages[idx]
        
        if self.transform:
            coherence_matrix = self.transform(coherence_matrix)
            class_label = self.transform(class_label)
            age = self.transform(age)
            
        return {'coherence_matrix': coherence_matrix, 'class_label': class_label, 'age': age}