# Artifact preservation

Git tracks all discovered project source, configuration, text documentation,
compact audit/result records (up to 2 MiB each), authored scene geometry, and
one curated video of the latest actual trial. Original bytes are preserved;
line endings are not normalized because existing receipts hash these files.

Bulk trajectories, arrays, learned checkpoints, caches, logs, generated HTML
reports, most media, archives, and oversized records are excluded from Git.
They are **not deleted or moved**. The import inventory includes their paths,
sizes, exclusion reasons, and SHA256 checksums. Symlinks are recorded as links;
external targets are not followed or uploaded implicitly.

## Existing storage

- Mac evidence: `/Users/kadir/humanoid-sitting-control`
- Ubuntu project and outputs: `/home/kadir/humanoid-sitting`
- Runtime data: `/data/humanoid-sitting-perception`

The Mac inventory covers the control workspace. The Ubuntu source inventory
covers the selected project-owned source roots, including baseline directories.
It does not claim to hash every external dataset, runtime output, simulator
dependency, or user file. Historical experiment manifests retain references to
those external inputs. GitHub is a source and compact-evidence backup, not a
complete off-machine backup of all experimental data.

Excluded files can be identified in `repository_metadata/artifact_inventory.tsv`.
Ubuntu source provenance is in `repository_metadata/ubuntu_source_inventory.tsv`.
The latest display clip is deliberately retained in Git because it is small;
its receipt identifies the saved trajectory that generated it.

No dependency license is replaced and no project-wide license is asserted by
this import. Existing attribution and license files remain in their source trees.
