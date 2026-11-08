type SearchCardProps = {
  title: string
  excerpt: string
  source: string
}

/** Small presentational example used by the synthetic file search corpus. */
export function SearchCard({ title, excerpt, source }: SearchCardProps) {
  return (
    <article aria-label={`Search result: ${title}`}>
      <h2>{title}</h2>
      <p>{excerpt}</p>
      <small>Source: {source}</small>
    </article>
  )
}
