# train.py

from options.train_options import TrainOptions
from models import Generator, Discriminator, save_checkpoint, load_checkpoint, cycle_loss, kl_divergence, r2_loss, loss_fn
from data import CSVDatasetCycle
from util.utils import get_full_gene_list, get_batch_label_list
import torch
from torch.utils.data import DataLoader
import itertools
import os
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import pandas as pd

def create_dataloaders(opt):
    """Create source and target dataloaders."""
    source_dataset = CSVDatasetCycle(opt.feature_path)
    target_dataset = CSVDatasetCycle(opt.label_path)
    source_loader = DataLoader(source_dataset, batch_size=opt.batch_size, shuffle=False, drop_last=True)
    target_loader = DataLoader(target_dataset, batch_size=opt.batch_size, shuffle=False, drop_last=True)
    return source_loader, target_loader

def create_models(opt, gene_size, num_perturbed_gene, num_batch):
    """Create generator and discriminator models."""
    G = Generator(
        gene_size=gene_size, 
        num_perturbed_gene=num_perturbed_gene, 
        num_batch=num_batch,
        num_res_blocks=opt.num_res_blocks,
        use_attention=opt.use_attention,
        use_embedding=opt.use_embedding,
        use_batch_norm=opt.use_batch_norm,
        use_dropout=opt.use_dropout
    ).to(opt.device)
    F = Generator(
        gene_size=gene_size, 
        num_perturbed_gene=num_perturbed_gene, 
        num_batch=num_batch,
        num_res_blocks=opt.num_res_blocks,
        use_attention=opt.use_attention,
        use_embedding=opt.use_embedding,
        use_batch_norm=opt.use_batch_norm,
        use_dropout=opt.use_dropout
    ).to(opt.device)
    D_X = Discriminator(gene_size=gene_size, num_perturbed_gene=num_perturbed_gene, num_batch=num_batch).to(opt.device)
    D_Y = Discriminator(gene_size=gene_size, num_perturbed_gene=num_perturbed_gene, num_batch=num_batch).to(opt.device)
    return G, F, D_X, D_Y

def create_optimizers(G, F, D_X, D_Y, opt):
    """Create optimizers and schedulers."""
    optimizer_G = Adam(itertools.chain(G.parameters(), F.parameters()), lr=opt.learning_rate, betas=tuple(opt.betas))
    optimizer_D_X = Adam(D_X.parameters(), lr=opt.learning_rate, betas=tuple(opt.betas))
    optimizer_D_Y = Adam(D_Y.parameters(), lr=opt.learning_rate, betas=tuple(opt.betas))
    scheduler_G = StepLR(optimizer_G, step_size=10, gamma=0.5)
    scheduler_D_X = StepLR(optimizer_D_X, step_size=10, gamma=0.5)
    scheduler_D_Y = StepLR(optimizer_D_Y, step_size=10, gamma=0.5)
    return optimizer_G, optimizer_D_X, optimizer_D_Y, scheduler_G, scheduler_D_X, scheduler_D_Y

def train(opt):
    source_loader, target_loader = create_dataloaders(opt)

    full_gene_list = get_full_gene_list()
    batch_label_list = get_batch_label_list()
    gene_size = len(full_gene_list)
    num_perturbed_gene = gene_size
    num_batch = len(batch_label_list)
    G, F, D_X, D_Y = create_models(opt, gene_size, num_perturbed_gene, num_batch)
    optimizer_G, optimizer_D_X, optimizer_D_Y, scheduler_G, scheduler_D_X, scheduler_D_Y = create_optimizers(G, F, D_X, D_Y, opt)
    writer = SummaryWriter(log_dir=os.path.join(opt.model_output_path, 'logs'))
    best_val_loss = float('inf')
    trigger_times = 0
    losses_G, losses_cycle_X, losses_cycle_Y, losses_D_X, losses_D_Y = [], [], [], [], []
    start_epoch = 0
    if opt.resume:
        G, F, (start_epoch, D_X, D_Y, optimizer_G, optimizer_D_X, optimizer_D_Y, \
               scheduler_G, scheduler_D_X, scheduler_D_Y, best_val_loss, trigger_times) = load_checkpoint(opt.model_path, G, F, D_X, D_Y, optimizer_G, optimizer_D_X, optimizer_D_Y, \
               scheduler_G, scheduler_D_X, scheduler_D_Y, best_val_loss, trigger_times)
    for epoch in range(start_epoch, opt.num_epochs):
        print(f"Epoch {epoch+1}/{opt.num_epochs}")
        progress_bar = tqdm(enumerate(zip(source_loader, target_loader)), total=len(source_loader))
        for i, (real_X, real_Y) in progress_bar:
            real_X = real_X.to(opt.device)
            real_Y = real_Y.to(opt.device)
            batch_size = real_X.size(0)

            # 拆分real_X
            x_X = real_X[:, :gene_size]
            perturbed_gene_X = real_X[:, gene_size:gene_size+num_perturbed_gene]
            batch_X = real_X[:, gene_size+num_perturbed_gene:]

            # 拆分real_Y
            x_Y = real_Y[:, :gene_size]
            perturbed_gene_Y = real_Y[:, gene_size:gene_size+num_perturbed_gene]
            batch_Y = real_Y[:, gene_size+num_perturbed_gene:]

            # ------------------
            #  Train Generators
            # ------------------
            for param in D_X.parameters():
                param.requires_grad = False
            for param in D_Y.parameters():
                param.requires_grad = False

            # GAN loss
            fake_Y_exp, fake_Y_p, fake_Y_b, latents_G = G(x_X, perturbed_gene_X, batch_X)
            mu_exp_g, logvar_exp_g, mu_ko_g, logvar_ko_g, mu_noise_g, logvar_noise_g = latents_G
            pred_fake = D_Y(fake_Y_exp, perturbed_gene_X, batch_X)
            loss_GAN_XY = loss_fn(pred_fake.view(-1), torch.ones_like(pred_fake).view(-1))  

            fake_X_exp, fake_X_p, fake_X_b, latents_F = F(x_Y, perturbed_gene_Y, batch_Y)
            mu_exp_f, logvar_exp_f, mu_ko_f, logvar_ko_f, mu_noise_f, logvar_noise_f = latents_F
            pred_fake = D_X(fake_X_exp, perturbed_gene_Y, batch_Y)
            loss_GAN_YX = loss_fn(pred_fake.view(-1), torch.ones_like(pred_fake).view(-1))  

            # Cycle loss
            recov_X_exp, recov_X_p, recov_X_b, _ = F(fake_Y_exp, perturbed_gene_X, batch_X)
            loss_cycle_X = cycle_loss(recov_X_exp, x_X, recov_X_p, perturbed_gene_X, recov_X_b, batch_X)

            recov_Y_exp, recov_Y_p, recov_Y_b, _ = G(fake_X_exp, perturbed_gene_Y, batch_Y)
            loss_cycle_Y = cycle_loss(recov_Y_exp, x_Y, recov_Y_p, perturbed_gene_Y, recov_Y_b, batch_Y)
            r2_val = r2_loss(x_X.detach().cpu().numpy(), recov_X_exp.detach().cpu().numpy())


            # KL loss 
            kl_g_exp   = kl_divergence(mu_exp_g,   logvar_exp_g,   reduction='mean')
            kl_g_ko    = kl_divergence(mu_ko_g,    logvar_ko_g,    reduction='mean')
            kl_g_noise = kl_divergence(mu_noise_g, logvar_noise_g, reduction='mean')
            kl_G = kl_g_exp + kl_g_ko + kl_g_noise  # G的总KL

            kl_f_exp   = kl_divergence(mu_exp_f,   logvar_exp_f,   reduction='mean')
            kl_f_ko    = kl_divergence(mu_ko_f,    logvar_ko_f,    reduction='mean')
            kl_f_noise = kl_divergence(mu_noise_f, logvar_noise_f, reduction='mean')
            kl_F = kl_f_exp + kl_f_ko + kl_f_noise  # F的总KL

            # Total loss
            loss_G = opt.lambda_gan * (loss_GAN_XY + loss_GAN_YX) + opt.lambda_cycle * (loss_cycle_X + loss_cycle_Y) + opt.lambda_kl * (kl_G + kl_F)

            optimizer_G.zero_grad()
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(itertools.chain(G.parameters(), F.parameters()), max_norm=1)
            optimizer_G.step()

            for param in D_X.parameters():
                param.requires_grad = True
            for param in D_Y.parameters():
                param.requires_grad = True

            # -----------------------
            #  Train Discriminator X
            # -----------------------
            pred_real = D_X(x_X, perturbed_gene_X, batch_X)
            loss_D_real = loss_fn(pred_real.view(-1), torch.ones_like(pred_real).view(-1))  
            pred_fake = D_X(fake_X_exp.detach(), perturbed_gene_Y, batch_Y)
            loss_D_fake = loss_fn(pred_fake.view(-1), torch.zeros_like(pred_fake).view(-1))  

            eps = torch.rand(batch_size, 1).to(opt.device)
            x_hat = eps * x_X + (1 - eps) * fake_X_exp.detach()
            x_hat.requires_grad_(True)
            pred_hat = D_X(x_hat, perturbed_gene_Y, batch_Y)
            gradients = torch.autograd.grad(outputs=pred_hat, inputs=x_hat,
                                grad_outputs=torch.ones(pred_hat.size()).to(opt.device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient_penalty = 10 * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            loss_D_X = (loss_D_real + loss_D_fake) * 0.5 + gradient_penalty
            optimizer_D_X.zero_grad()
            loss_D_X.backward()
            optimizer_D_X.step()

            # -----------------------
            #  Train Discriminator Y
            # -----------------------
            pred_real = D_Y(x_Y, perturbed_gene_Y, batch_Y)
            loss_D_real = loss_fn(pred_real.view(-1), torch.ones_like(pred_real).view(-1))  
            pred_fake = D_Y(fake_Y_exp.detach(), perturbed_gene_X, batch_X)
            loss_D_fake = loss_fn(pred_fake.view(-1), torch.zeros_like(pred_fake).view(-1))
            eps = torch.rand(batch_size, 1).to(opt.device)
            y_hat = eps * x_Y + (1 - eps) * fake_Y_exp.detach()
            y_hat.requires_grad_(True)
            pred_hat = D_Y(y_hat, perturbed_gene_X, batch_X)
            gradients = torch.autograd.grad(outputs=pred_hat, inputs=y_hat,
                                grad_outputs=torch.ones(pred_hat.size()).to(opt.device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient_penalty = 10 * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            loss_D_Y = (loss_D_real + loss_D_fake) * 0.5 + gradient_penalty
            optimizer_D_Y.zero_grad()
            loss_D_Y.backward()
            optimizer_D_Y.step()
            
            progress_bar.set_description(f'Loss: {loss_G:.4f}; Gene_r2: {r2_val:.4f}; Cycle: {(loss_cycle_X + loss_cycle_Y):.4f}; GAN: {(loss_GAN_XY + loss_GAN_YX):.4f}; D_X: {loss_D_X:.4f}; D_Y: {loss_D_Y:.4f}')

            writer.add_scalar('Loss/G', loss_G.item(), epoch * len(source_loader) + i)
            writer.add_scalar('Loss/D_X', loss_D_X.item(), epoch * len(source_loader) + i)
            writer.add_scalar('Loss/D_Y', loss_D_Y.item(), epoch * len(source_loader) + i)
            writer.add_scalar('Loss/cycle_X', loss_cycle_X.item(), epoch * len(source_loader) + i)
            writer.add_scalar('Loss/cycle_Y', loss_cycle_Y.item(), epoch * len(source_loader) + i)

            losses_G.append(loss_G.item())
            losses_cycle_X.append(loss_cycle_X.item())
            losses_cycle_Y.append(loss_cycle_Y.item())
            losses_D_X.append(loss_D_X.item())
            losses_D_Y.append(loss_D_Y.item())
        
        scheduler_G.step()
        scheduler_D_X.step()
        scheduler_D_Y.step()
        
        if loss_G < best_val_loss:
            best_val_loss = loss_G
            trigger_times = 0
            save_checkpoint(epoch, G, F, D_X, D_Y, optimizer_G, optimizer_D_X, optimizer_D_Y,
                            scheduler_G, scheduler_D_X, scheduler_D_Y, best_val_loss, trigger_times,
                            os.path.join(opt.model_output_path, f'{opt.name}_best_checkpoint.pth'))
        else:
            trigger_times += 1
            if trigger_times >= opt.patience:
                print('Early stopping!')
                break
        
        save_checkpoint(epoch, G, F, D_X, D_Y, optimizer_G, optimizer_D_X, optimizer_D_Y,
                        scheduler_G, scheduler_D_X, scheduler_D_Y, best_val_loss, trigger_times,
                        os.path.join(opt.model_output_path, f'{opt.name}_checkpoint_{epoch}.pth'))

        df = pd.DataFrame({
            'loss_G': losses_G,
            'loss_cycle_X': losses_cycle_X,
            'loss_cycle_Y': losses_cycle_Y,
            'loss_D_X': losses_D_X,
            'loss_D_Y': losses_D_Y
        })
        df.to_csv(os.path.join(opt.model_output_path, f'{opt.name}_loss.csv'), index = False)
    writer.close()

if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(opt)