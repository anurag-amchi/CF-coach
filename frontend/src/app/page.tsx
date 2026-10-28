// Slice 1 (decision #51): a placeholder home page that proves Next.js + Tailwind run here.
// The real start screen (enter a handle -> sync -> profile) replaces it in a later slice.
export default function Home() {
  return (
    <main className="mx-auto flex min-h-screen max-w-xl flex-col justify-center gap-4 px-4">
      <h1 className="text-3xl font-semibold tracking-tight">CF Coach</h1>
      <p className="text-zinc-600 dark:text-zinc-400">
        A practice coach for Codeforces: it reads your submissions, measures your skill per
        tag, and builds a weekly plan that climbs from easy to hard.
      </p>
    </main>
  );
}
