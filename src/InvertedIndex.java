import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.regex.Pattern;

import org.apache.commons.validator.routines.UrlValidator;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.StringUtils;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

public class InvertedIndex extends Configured implements Tool {

	private static final Logger LOG = Logger.getLogger(InvertedIndex.class);
	private static final String WORD_MARKER = "W"; // Used to indicate that a string is a word.
	private static final String URL_MARKER = "U"; // Used to indicate that a string is a URL.
	private static final String WORD_TYPE_SEPERATOR = "#";

	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new InvertedIndex(), args);
		System.exit(res);
	}

	public int run(String[] args) throws Exception {
		Job job = Job.getInstance(getConf(), "invertedindex");
		for (int i = 0; i < args.length; i += 1) {
			if ("-skip".equals(args[i])) {
				job.getConfiguration().setBoolean("wordcount.skip.patterns", true);
				i += 1;
				job.addCacheFile(new Path(args[i]).toUri());
				// this demonstrates logging
				LOG.info("Added file to the distributed cache: " + args[i]);
			}
		}

		job.setJarByClass(this.getClass());
		// Use TextInputFormat, the default unless job.setInputFormatClass is used
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		job.setMapperClass(InvertedIndexMap.class);
		job.setReducerClass(InvertedIndexReduce.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

		return job.waitForCompletion(true) ? 0 : 1;
	}

	public static class InvertedIndexMap extends Mapper<LongWritable, Text, Text, Text> {
		private final static Text one = new Text("1");
		private boolean caseSensitive = false;
		private Set<String> patternsToSkip = new HashSet<String>();

		// Create a regular expression pattern used to parse each line of input text on word boundaries. Word boundaries include spaces, tabs, and punctuation.
		private static final Pattern WORD_BOUNDARY = Pattern.compile(" ");

		@Override
		protected void setup(Mapper<LongWritable, Text, Text, Text>.Context context) throws IOException, InterruptedException {
			if (context.getInputSplit() instanceof FileSplit) {
				((FileSplit) context.getInputSplit()).getPath().toString();
			} else {
				context.getInputSplit().toString();
			}
			Configuration config = context.getConfiguration();
			this.caseSensitive = config.getBoolean("wordcount.case.sensitive", false);
			if (config.getBoolean("wordcount.skip.patterns", false)) {
				URI[] localPaths = context.getCacheFiles();
				parseSkipFile(localPaths[0]);
			}
		}

		private void parseSkipFile(URI patternsURI) {
			LOG.info("Added file to the distributed cache: " + patternsURI);
			
			BufferedReader fis = null;
			try {
				fis = new BufferedReader(new FileReader(new File(patternsURI.getPath()).getName()));
				String pattern;
				while ((pattern = fis.readLine()) != null) {
					patternsToSkip.add(pattern);
				}
			} catch (IOException ioe) {
				LOG.error("Caught exception while parsing the cached file " + patternsURI + ".", ioe);
			} finally {
				if (fis != null) {
					try {
						fis.close();
					} catch (IOException e) {
						LOG.error("Failed to close pattern file. ", e);
					}
				}
			}
		}

		@Override
		public void map(LongWritable offset, Text lineText, Context context) throws IOException, InterruptedException {
			String line = lineText.toString();
			if (!caseSensitive) {
				line = line.toLowerCase();
			}

			// Get the site URL that is always the first word of the line.
			int firstWordIdx = line.indexOf(" ");
			String ownerSite = line.substring(0, firstWordIdx);
			Text site = new Text(ownerSite);
			line = line.replaceFirst(ownerSite, "");
			
			String [] words = WORD_BOUNDARY.split(line);			
			for (int i = 1; i < words.length; i++) {
				String word = words[i]; // Get the current word to process.
				if (word.isEmpty() || patternsToSkip.contains(word)) {
					continue;
				}

				Text currentWord = null;
				UrlValidator urlValidator = new UrlValidator();
				if (urlValidator.isValid(word)) {
					// The word is a URL. Mark it as a one and count its occurrence.
					currentWord = new Text(word + WORD_TYPE_SEPERATOR + URL_MARKER);
					context.write(currentWord, one);
				} else {
					// We have an actual word. Mark it as one and collect the current site.
					currentWord = new Text(word + WORD_TYPE_SEPERATOR + WORD_MARKER);
					context.write(currentWord, site);
				}
			}
		}
	}

	public static class InvertedIndexReduce extends Reducer<Text, Text, Text, Text> {

		@Override
		public void reduce(Text word, Iterable<Text> occurences, Context context) throws IOException, InterruptedException {
			String [] wordParts = word.toString().split(WORD_TYPE_SEPERATOR);
			Text result = null;			
			String wordType = wordParts[1];
			switch (wordType) {
			case URL_MARKER:
				result = collectURLOccurences(occurences);
				break;
			case WORD_MARKER:
				result = collectWordSites(occurences);				
				break;
			};

			Text currWord = new Text(wordParts[0]);
			context.write(currWord, result);
		}

		private Text collectURLOccurences(Iterable<Text> occurences) {
			Integer numOfOccurences = 0;
			for (Text count : occurences) {
				int num = Integer.valueOf(count.toString());
				numOfOccurences += num;
			}

			Text urlOccurences = new Text(numOfOccurences.toString());
			return urlOccurences;
		}

		private Text collectWordSites(Iterable<Text> occurences) {
			Set<Text> sites = new HashSet<Text>();
			for (Text site : occurences) {
				sites.add(site);
			}

			Text joindSites = new Text(join(sites, ","));
			return joindSites;
		}

		private static <T> String join(final Collection<T> objs, final String delimiter) {
			if (objs == null || objs.isEmpty())
				return "";

			Iterator<T> iter = objs.iterator();
			if (!iter.hasNext())
				return "";

			StringBuffer buffer = new StringBuffer(String.valueOf(iter.next()));
			while (iter.hasNext())
				buffer.append(delimiter).append(String.valueOf(iter.next()));

			return buffer.toString();
		}
	}
}