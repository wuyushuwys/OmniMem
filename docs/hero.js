// Keep decorative playback independent of the demonstration gallery.
(() => {
  const video = document.getElementById('hero-video');
  if (!video) return;

  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let wantsPlayback = !motion.matches;
  let inView = !('IntersectionObserver' in window);
  video.muted = true;

  async function syncPlayback() {
    if (!wantsPlayback || !inView || document.hidden) {
      video.pause();
      return;
    }
    try {
      await video.play();
      // Scrolling or the system motion preference may change while loading.
      if (!wantsPlayback || !inView || document.hidden) video.pause();
    } catch (_) {
      // The poster remains visible if autoplay is unavailable.
    }
  }

  document.addEventListener('visibilitychange', syncPlayback);
  motion.addEventListener('change', () => {
    wantsPlayback = !motion.matches;
    syncPlayback();
  });

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(([entry]) => {
      inView = entry.isIntersecting;
      syncPlayback();
    }, { threshold: 0 });
    observer.observe(video.closest('.hero'));
  }

  syncPlayback();
})();
