/**
 * AI-Powered Heavy Rainfall and Flood Early-Warning System - Atmospheric Rain & Lightning Canvas Engine
 * Cinematic heavy rain particles and subtle ambient lightning flashes
 */

class StormEngine {
  constructor(canvasId, options = {}) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');

    this.options = {
      particleCount: options.particleCount || 140,
      angle: options.angle || 0.22, // Slight diagonal tilt
      speed: options.speed || 18,
      lightningEnabled: options.lightningEnabled !== false,
      color: options.color || 'rgba(180, 220, 255, 0.45)',
      splashColor: options.splashColor || 'rgba(160, 215, 255, 0.35)',
      ...options
    };

    this.particles = [];
    this.splashes = [];
    this.width = 0;
    this.height = 0;
    this.isRunning = false;
    this.animId = null;
    this.lightningAlpha = 0;
    this.nextLightningTime = Date.now() + 4000 + Math.random() * 5000;

    this.init();
  }

  init() {
    this.resize();
    window.addEventListener('resize', () => this.resize());

    // Pause when tab not visible to preserve GPU
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        this.stop();
      } else {
        this.start();
      }
    });

    this.populate();
    this.start();
  }

  resize() {
    if (!this.canvas) return;
    const parent = this.canvas.parentElement;
    this.width = this.canvas.width = parent ? parent.clientWidth : window.innerWidth;
    this.height = this.canvas.height = parent ? parent.clientHeight : window.innerHeight;
  }

  populate() {
    this.particles = [];
    for (let i = 0; i < this.options.particleCount; i++) {
      this.particles.push({
        x: Math.random() * (this.width + 300) - 150,
        y: Math.random() * this.height,
        len: Math.random() * 22 + 12,
        speed: (Math.random() * 0.7 + 0.8) * this.options.speed,
        opacity: Math.random() * 0.45 + 0.25,
        thickness: Math.random() * 1.2 + 0.6
      });
    }
  }

  createSplash(x, y) {
    if (this.splashes.length > 50) return;
    const count = Math.floor(Math.random() * 3) + 2;
    for (let i = 0; i < count; i++) {
      this.splashes.push({
        x: x,
        y: y,
        vx: (Math.random() - 0.5) * 4,
        vy: -Math.random() * 3 - 1,
        radius: Math.random() * 1.8 + 0.8,
        alpha: 0.6,
        life: 0,
        maxLife: Math.floor(Math.random() * 10) + 8
      });
    }
  }

  update() {
    const now = Date.now();

    // Ambient lightning trigger
    if (this.options.lightningEnabled) {
      if (now > this.nextLightningTime) {
        this.lightningAlpha = Math.random() * 0.28 + 0.15;
        this.nextLightningTime = now + 6000 + Math.random() * 9000;

        // Secondary flicker
        setTimeout(() => {
          this.lightningAlpha = Math.random() * 0.22 + 0.1;
        }, 120);
      } else if (this.lightningAlpha > 0) {
        this.lightningAlpha -= 0.018;
        if (this.lightningAlpha < 0) this.lightningAlpha = 0;
      }
    }

    // Update raindrops
    const dx = Math.sin(this.options.angle);
    const dy = Math.cos(this.options.angle);

    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];
      p.x += dx * p.speed;
      p.y += dy * p.speed;

      // Reset when off screen
      if (p.y > this.height) {
        this.createSplash(p.x, this.height - 2);
        p.y = -p.len - Math.random() * 50;
        p.x = Math.random() * (this.width + 300) - 150;
      }
      if (p.x > this.width + 100) {
        p.x = -50;
      }
    }

    // Update splash particles
    for (let i = this.splashes.length - 1; i >= 0; i--) {
      const s = this.splashes[i];
      s.x += s.vx;
      s.y += s.vy;
      s.vy += 0.22; // Gravity
      s.life++;
      s.alpha = 0.6 * (1 - s.life / s.maxLife);
      if (s.life >= s.maxLife) {
        this.splashes.splice(i, 1);
      }
    }
  }

  draw() {
    this.ctx.clearRect(0, 0, this.width, this.height);

    // Draw ambient lightning illumination
    if (this.lightningAlpha > 0.01) {
      this.ctx.fillStyle = `rgba(180, 230, 255, ${this.lightningAlpha})`;
      this.ctx.fillRect(0, 0, this.width, this.height);
    }

    // Draw raindrops
    const dx = Math.sin(this.options.angle);
    const dy = Math.cos(this.options.angle);

    this.ctx.lineCap = 'round';
    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];
      this.ctx.beginPath();
      this.ctx.lineWidth = p.thickness;
      this.ctx.strokeStyle = `rgba(190, 230, 255, ${p.opacity})`;
      this.ctx.moveTo(p.x, p.y);
      this.ctx.lineTo(p.x - dx * p.len, p.y - dy * p.len);
      this.ctx.stroke();
    }

    // Draw splashes
    for (let i = 0; i < this.splashes.length; i++) {
      const s = this.splashes[i];
      this.ctx.beginPath();
      this.ctx.fillStyle = `rgba(190, 230, 255, ${s.alpha})`;
      this.ctx.arc(s.x, s.y, s.radius, 0, Math.PI * 2);
      this.ctx.fill();
    }
  }

  loop() {
    if (!this.isRunning) return;
    this.update();
    this.draw();
    this.animId = requestAnimationFrame(() => this.loop());
  }

  start() {
    if (this.isRunning) return;
    this.isRunning = true;
    this.loop();
  }

  stop() {
    this.isRunning = false;
    if (this.animId) {
      cancelAnimationFrame(this.animId);
      this.animId = null;
    }
  }

  toggle() {
    if (this.isRunning) {
      this.stop();
      this.ctx.clearRect(0, 0, this.width, this.height);
      return false;
    } else {
      this.start();
      return true;
    }
  }
}

window.StormEngine = StormEngine;
