from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, prompt_path, switch_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if switch_prompt_path is not None:
            with open(switch_prompt_path, encoding="utf-8") as f:
                self.switch_prompt_list = [line.rstrip() for line in f]
            assert len(self.switch_prompt_list) == len(self.prompt_list)
        else:
            self.switch_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.switch_prompt_list is not None:
            batch["switch_prompts"] = self.switch_prompt_list[idx]
        return batch
