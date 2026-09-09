"""CRF chuỗi tuyến tính có ràng buộc + biên duyên (spec §7.1, §9.1).

Tự cài vì máy này không có torchcrf. Ngoài log-likelihood và Viterbi, lớp này
BẮT BUỘC cung cấp biên duyên forward-backward: spec §9.1 nói rõ không được lấy
log-likelihood của cả chuỗi làm độ tin cậy của một span.
"""

import torch
import torch.nn as nn

from .crf_constraints import build_constraint_tensors


class ConstrainedCRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        t, s, e = build_constraint_tensors()
        self.register_buffer("trans_penalty", t)
        self.register_buffer("start_penalty", s)
        self.register_buffer("end_penalty", e)

    # --- tham số đã cộng ràng buộc ---
    def _params(self):
        return (self.transitions + self.trans_penalty,
                self.start_transitions + self.start_penalty,
                self.end_transitions + self.end_penalty)

    def _score(self, emissions, tags, mask):
        """Điểm của chuỗi nhãn cho trước. emissions [B,T,K], tags [B,T], mask [B,T]."""
        trans, start, end = self._params()
        batch, seq_len, _ = emissions.shape
        mask = mask.to(emissions.dtype)
        score = start[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for i in range(1, seq_len):
            step = trans[tags[:, i - 1], tags[:, i]] + \
                emissions[:, i].gather(1, tags[:, i:i + 1]).squeeze(1)
            score = score + step * mask[:, i]
        lengths = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, lengths.unsqueeze(1)).squeeze(1)
        return score + end[last_tags]

    def _forward_alpha(self, emissions, mask):
        """alpha[b,i,k] = log tổng điểm mọi đường tới nhãn k tại bước i."""
        trans, start, _ = self._params()
        batch, seq_len, num_tags = emissions.shape
        alphas = [start.unsqueeze(0) + emissions[:, 0]]
        for i in range(1, seq_len):
            broadcast = alphas[-1].unsqueeze(2) + trans.unsqueeze(0) + \
                emissions[:, i].unsqueeze(1)
            nxt = torch.logsumexp(broadcast, dim=1)
            m = mask[:, i].unsqueeze(1).to(emissions.dtype)
            alphas.append(nxt * m + alphas[-1] * (1 - m))
        return torch.stack(alphas, dim=1)

    def _backward_beta(self, emissions, mask):
        trans, _, end = self._params()
        batch, seq_len, num_tags = emissions.shape
        betas = [None] * seq_len
        lengths = mask.long().sum(dim=1)
        last = torch.zeros(batch, num_tags, dtype=emissions.dtype, device=emissions.device)
        betas[seq_len - 1] = end.unsqueeze(0).expand(batch, -1).clone()
        for i in range(seq_len - 2, -1, -1):
            broadcast = trans.unsqueeze(0) + emissions[:, i + 1].unsqueeze(1) + \
                betas[i + 1].unsqueeze(1)
            nxt = torch.logsumexp(broadcast, dim=2)
            m = mask[:, i + 1].unsqueeze(1).to(emissions.dtype)
            # bước bị pad: giữ nguyên beta của bước cuối thực sự
            keep = end.unsqueeze(0).expand(batch, -1)
            betas[i] = nxt * m + keep * (1 - m)
        return torch.stack(betas, dim=1)

    def partition(self, emissions, mask):
        _, _, end = self._params()
        alphas = self._forward_alpha(emissions, mask)
        lengths = mask.long().sum(dim=1) - 1
        idx = lengths.view(-1, 1, 1).expand(-1, 1, emissions.shape[2])
        last_alpha = alphas.gather(1, idx).squeeze(1)
        return torch.logsumexp(last_alpha + end.unsqueeze(0), dim=1)

    def neg_log_likelihood(self, emissions, tags, mask, reduction="mean"):
        nll = self.partition(emissions, mask) - self._score(emissions, tags, mask)
        if reduction == "sum":
            return nll.sum()
        if reduction == "none":
            return nll
        return nll.mean()

    def marginals(self, emissions, mask):
        """P(nhãn tại bước i | quan sát) — nền cho boundary_confidence (spec §9.1)."""
        alphas = self._forward_alpha(emissions, mask)
        betas = self._backward_beta(emissions, mask)
        logZ = self.partition(emissions, mask).view(-1, 1, 1)
        return torch.exp(alphas + betas - logZ)

    def decode(self, emissions, mask):
        """Viterbi có ràng buộc. Trả list các list nhãn (đã bỏ phần pad)."""
        trans, start, end = self._params()
        batch, seq_len, num_tags = emissions.shape
        score = start.unsqueeze(0) + emissions[:, 0]
        history = []
        for i in range(1, seq_len):
            broadcast = score.unsqueeze(2) + trans.unsqueeze(0) + emissions[:, i].unsqueeze(1)
            best, idx = broadcast.max(dim=1)
            m = mask[:, i].unsqueeze(1).to(emissions.dtype)
            score = best * m + score * (1 - m)
            history.append(idx)
        score = score + end.unsqueeze(0)

        lengths = mask.long().sum(dim=1)
        paths = []
        for b in range(batch):
            length = int(lengths[b].item())
            best_tag = int(score[b].argmax().item())
            path = [best_tag]
            for i in range(length - 2, -1, -1):
                best_tag = int(history[i][b][best_tag].item())
                path.append(best_tag)
            paths.append(list(reversed(path)))
        return paths
