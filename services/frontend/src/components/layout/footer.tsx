export function Footer() {
  const year = new Date().getFullYear();
  return (
    <footer className="border-t py-6 md:py-0">
      <div className="container flex flex-col items-center justify-between gap-4 md:h-14 md:flex-row">
        <p className="text-sm text-muted-foreground">
          © {year} Auspex — sports prediction analytics
        </p>
        <p className="text-xs text-muted-foreground">
          For entertainment purposes only. Please gamble responsibly.
        </p>
      </div>
    </footer>
  );
}
