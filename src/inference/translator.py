import torch
import torch.nn.functional as F

class Translator:
    def __init__(self, model, src_tokenizer, tgt_tokenizer, device, max_gen_len=50, beam_size=1):
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = device
        self.max_gen_len = max_gen_len
        self.beam_size = beam_size
        self.model.eval()

    def greedy_decode(self, src_sentence: str) -> str:
        src_ids = self.src_tokenizer.encode(src_sentence, add_start=False, add_end=False).unsqueeze(0).to(self.device)
        # Start with <START> token
        tgt_ids = torch.tensor([[self.tgt_tokenizer.start_id]], device=self.device)
        for _ in range(self.max_gen_len):
            with torch.no_grad():
                logits = self.model(src_ids, tgt_ids, src_add_start=False, src_add_end=False,
                                    tgt_add_start=False, tgt_add_end=False)  # tgt_ids already includes start
            next_logits = logits[0, -1, :]
            next_token = next_logits.argmax(dim=-1).unsqueeze(0).unsqueeze(0)
            if next_token.item() == self.tgt_tokenizer.end_id:
                break
            tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
        return self.tgt_tokenizer.decode(tgt_ids[0], skip_special=True)

    def translate(self, src_sentence: str) -> str:
        if self.beam_size == 1:
            return self.greedy_decode(src_sentence)
        else:
            # Placeholder for beam search (optional)
            raise NotImplementedError("Beam search not yet implemented")