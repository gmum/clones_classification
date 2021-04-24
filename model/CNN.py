import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.args = args
        self.L = self.args.L
        self.D = self.args.D
        self.K = self.args.K

        if self.args.loc_info:
            self.add = 2
        else:
            self.add = 0

        if self.args.dataset_name == 'bacteria':
            input_dim = 512
        else:
            raise ValueError('Expected bacteria dataset')

        if self.args['operator'] == 'att':
            self.conv1x1 = nn.Conv1d(input_dim, input_dim // 2, 1)
            input_dim = input_dim // 2
        if self.args.self_att:
            self.self_att = SelfAttention(input_dim, self.args)
        if self.args['operator'] == 'att':
            self.attention = nn.Sequential(  # first layer
                nn.Linear(input_dim, self.D),
                nn.Tanh(),
                # second layer
                nn.Linear(self.D, self.K)
                # outputs A: NxK
            )

            torch.nn.init.xavier_uniform_(self.attention[0].weight)
            self.attention[0].bias.data.zero_()
            torch.nn.init.xavier_uniform_(self.attention[2].weight)
            self.attention[2].bias.data.zero_()

            self.classifier = nn.Sequential(
                nn.Linear(input_dim * self.K, self.args.output_dim),
            )
        elif self.args['operator'] in ['mean', 'max']:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, self.args.output_dim),
            )

        torch.nn.init.xavier_uniform_(self.classifier[0].weight)
        self.classifier[0].bias.data.zero_()

    def forward(self, x):
        if self.args['dataset_name'] == 'bacteria':
            x = x.unsqueeze(1)
        if not self.args.out_loc:
            loc = x[:, 3:]
            x = x[:, :3]
        x = x.permute((0, 2, 1))
        if self.args['operator'] == 'att':
            H = self.conv1x1(x)
            H = H.mean(2)
        else:
            H = x
        if self.args['dataset_name'] == 'fungus':
            H = H.squeeze(0)
        H = H.view(-1, H.shape[1])
        gamma, gamma_kernel = (0, 0)
        if self.args.self_att:
            H, self_attention, gamma, gamma_kernel = self.self_att(H)
        # attention
        if self.args['operator'] == 'mean':
            M = H.mean(0)
        elif self.args['operator'] == 'max':
            M, _ = torch.max(H, 0)
        elif self.args['operator'] == 'att':
            A = self.attention(H)  # NxK
            A = torch.transpose(A, 1, 0)  # KxN

            z = F.softmax(A)  # softmax over N

            M = torch.mm(z, H)  # KxL

            M = M.view(1, -1)  # (K*L)x1
        # classification
        y_prob = self.classifier(M)
        if self.args['operator'] in ['mean', 'max']:
            y_prob = y_prob.unsqueeze(0)
        _, y_hat = torch.max(y_prob, 1)
        if self.args['operator'] in ['mean', 'max']:
            return y_prob, y_hat, [], [], gamma, gamma_kernel
        elif self.args.self_att:
            return y_prob, y_hat, z, (A, self_attention), gamma, gamma_kernel
        else:
            return y_prob, y_hat, z, A, gamma, gamma_kernel

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        # Y = Y.float()
        y_prob, y_hat, _, _, gamma, gamma_kernel = self.forward(X)
        error = 1. - y_hat.eq(Y).cpu().float().mean()
        return error, gamma, gamma_kernel

    def calculate_objective(self, X, Y):
        # Y = Y.float()
        y_prob, _, _, _, gamma, gamma_kernel = self.forward(X)
        loss = self.criterion(y_prob, Y.view(1))
        return loss, gamma, gamma_kernel


class SelfAttention(nn.Module):
    def __init__(self, in_dim, args):
        super(SelfAttention, self).__init__()
        self.args = args
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter((torch.ones(1)).cuda())
        self.gamma_in = nn.Parameter((torch.ones(1)).cuda())
        self.softmax = nn.Softmax(dim=-1)
        self.alfa = nn.Parameter((torch.ones(1)).cuda())
        self.gamma_att = nn.Parameter((torch.ones(1)).cuda())

    def forward(self, x):
        if self.args.loc_info:
            loc = x[:, -2:]
            x = x[:, :-2]

        x = x.view(1, x.shape[0], x.shape[1]).permute((0, 2, 1))
        bs, C, length = x.shape
        proj_query = self.query_conv(x).view(bs, -1, length).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(bs, -1, length)  # B X C x (*W*H)

        if self.args.att_gauss_spatial:
            proj = torch.zeros((length, length))
            if self.args.cuda:
                proj = proj.cuda()
            proj_query = proj_query.permute(0, 2, 1)
            for i in range(length):
                gauss = torch.pow(proj_query - proj_key[:, :, i].t(), 2).sum(dim=1)
                proj[:, i] = torch.exp(-F.relu(self.gamma_att) * gauss)
            energy = proj.view((1, length, length))
        elif self.args.att_inv_q_spatial:
            proj = torch.zeros((length, length))
            if self.args.cuda:
                proj = proj.cuda()
            proj_query = proj_query.permute(0, 2, 1)
            for i in range(length):
                gauss = torch.pow(proj_query - proj_key[:, :, i].t(), 2).sum(dim=1)
                proj[:, i] = 1 / (F.relu(self.gamma_att) * gauss + torch.ones(1).cuda())
            energy = proj.view((1, length, length))
        elif self.args.att_module:
            proj = torch.zeros((length, length))
            if self.args.cuda:
                proj = proj.cuda()
            proj_query = proj_query.permute(0, 2, 1)
            for i in range(length):
                proj[:, i] = (torch.abs(proj_query - proj_key[:, :, i].t()) -
                              torch.abs(proj_query) -
                              torch.abs(proj_key[:, :, i].t())).sum(dim=1)
            energy = proj.view((1, length, length))
        elif self.args.laplace_att:
            proj = torch.zeros((length, length))
            if self.args.cuda:
                proj = proj.cuda()
            proj_query = proj_query.permute(0, 2, 1)
            for i in range(length):
                proj[:, i] = (-torch.abs(proj_query - proj_key[:, :, i].t())).sum(dim=1)
            energy = proj.view((1, length, length))

        elif self.args.att_gauss_abnormal:
            proj = torch.zeros((length, length))
            if self.args.cuda:
                proj = proj.cuda()
            proj_query = proj_query.permute(0, 2, 1)
            for i in range(int(C // 8)):
                gauss = proj_query[0, i, :] - proj_key[0, i, :].view(-1, 1)
                proj += torch.exp(-F.relu(self.gamma_att) * torch.abs(torch.pow(gauss, 2)))
            energy = proj.view((1, length, length))

        elif self.args.att_inv_q_abnormal:
            proj = torch.zeros((length, length)).cuda()
            proj_query = proj_query.permute(0, 2, 1)
            for i in range(int(C // 8)):
                gauss = proj_query[0, i, :] - proj_key[0, i, :].view(-1, 1)
                proj += torch.exp(F.relu(1 / (torch.pow(gauss, 2) + torch.tensor(1).cuda())))
            energy = proj.view((1, length, length))

        else:
            energy = torch.bmm(proj_query, proj_key)  # transpose check

        if self.args.loc_info:
            if self.args.loc_gauss:
                loc_energy_x = torch.exp(
                    -F.relu(self.gamma_in) * torch.abs(torch.pow(loc[:, 0] - loc[:, 0].view(-1, 1), 2)))
                loc_energy_y = torch.exp(
                    -F.relu(self.gamma_in) * torch.abs(torch.pow(loc[:, 1] - loc[:, 1].view(-1, 1), 2)))
                energy_pos = self.alfa * (loc_energy_x + loc_energy_y)
                energy = energy + energy_pos
            elif self.args.loc_inv_q:
                loc_energy_x = torch.exp(
                    1 / (torch.abs(torch.pow(loc[:, 0] - loc[:, 0].view(-1, 1), 2) + torch.tensor(1).cuda())))
                loc_energy_y = torch.exp(
                    1 / (torch.abs(torch.pow(loc[:, 1] - loc[:, 1].view(-1, 1), 2) + torch.tensor(1).cuda())))
                energy_pos = self.alfa * loc_energy_x + loc_energy_y
                energy = energy + energy_pos

            elif self.args.loc_att:
                loc_proj = torch.zeros((length, length))
                if self.args.cuda:
                    loc_proj = loc_proj.cuda()
                rel_loc_x = loc[:, 0] - loc[:, 0].view(-1, 1)
                rel_loc_y = loc[:, 1] - loc[:, 1].view(-1, 1)
                for i in range(length):
                    rel_loc_at = torch.sum(proj_query[0] * rel_loc_x[:, i].view(-1) * rel_loc_y[i, :].view(-1), dim=0)
                    loc_proj[:, i] = rel_loc_at
                energy += loc_proj.view((1, length, length))

        attention = self.softmax(energy)  # BX (N) X (N)

        proj_value = self.value_conv(x).view(bs, -1, length)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(bs, C, length)

        out = self.gamma * out + x
        return out[0].permute(1, 0), attention, self.gamma, self.gamma_att
